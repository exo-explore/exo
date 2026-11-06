import numpy as np

from exo.shared.types.common import NodeId
from exo.shared.types.peripheral import (
    LogitsPayload,
    PeripheralCapability,
    SamplingConfig,
    TruncatedLogitsEntry,
)
from exo.worker.peripheral.gateway import EdgeStreamingGateway, IncrementalUtf8Decoder
from exo.worker.peripheral.grammar_masker import SimpleJsonGrammarMasker
from exo.worker.peripheral.sampler import PerTokenLogitsSampler


def test_peripheral_capability_classification() -> None:
    # Legacy machine with 2GB RAM and no hardware accelerator
    weak_node = PeripheralCapability(
        node_id=NodeId("legacy_laptop_001"),
        roles=["tokenization_gateway", "sampling_grammar_processor"],
        available_memory_megabytes=2048,
        has_hardware_accelerator=False,
    )
    assert weak_node.is_peripheral_specialist is True

    # High-end GPU server with 64GB VRAM
    gpu_node = PeripheralCapability(
        node_id=NodeId("gpu_server_001"),
        roles=["compute_heavy"],
        available_memory_megabytes=65536,
        has_hardware_accelerator=True,
    )
    assert gpu_node.is_peripheral_specialist is False


def test_sampler_greedy_decoding() -> None:
    logits = np.array([1.2, 5.8, 3.4, 0.1], dtype=np.float32)
    config = SamplingConfig(temperature=0.0)

    sampled_id, logprob = PerTokenLogitsSampler.sample_from_dense_logits(logits, config)
    assert sampled_id == 1
    assert logprob <= 0.0


def test_sampler_repetition_penalty() -> None:
    logits = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    # Penalize token 0 heavily
    penalized = PerTokenLogitsSampler.apply_repetition_penalty(
        logits, generated_tokens=[0], penalty=2.0
    )
    assert penalized[0] == 2.0  # 4.0 / 2.0
    assert penalized[1] == 4.0
    assert penalized[2] == 4.0


def test_sampler_top_k_and_top_p() -> None:
    logits = np.array([10.0, 5.0, 1.0, 0.5, -2.0], dtype=np.float32)

    top_k_masked = PerTokenLogitsSampler.apply_top_k(logits, top_k=2)
    assert top_k_masked[0] == 10.0
    assert top_k_masked[1] == 5.0
    assert np.isneginf(top_k_masked[2])
    assert np.isneginf(top_k_masked[3])

    top_p_masked = PerTokenLogitsSampler.apply_top_p(logits, top_p=0.9)
    # The dominant logit 10.0 accounts for >99% probability, so others should be masked
    assert top_p_masked[0] == 10.0
    assert np.isneginf(top_p_masked[2])


def test_sampler_sparse_payload_reconstruction() -> None:
    sparse_entries = [
        TruncatedLogitsEntry(token_index=5, logit_value=9.5),
        TruncatedLogitsEntry(token_index=12, logit_value=7.2),
    ]
    payload = LogitsPayload(
        sequence_id="seq-test-1",
        step=42,
        vocabulary_size=20,
        sparse_entries=sparse_entries,
    )
    dense = PerTokenLogitsSampler.reconstruct_dense_logits(payload)
    assert len(dense) == 20
    assert dense[5] == 9.5
    assert dense[12] == 7.2
    assert np.isneginf(dense[0])
    assert np.isneginf(dense[19])


def test_json_grammar_masker_transitions() -> None:
    vocab = {
        0: "{",
        1: "}",
        2: '"name"',
        3: ":",
        4: '"Exo"',
        5: ",",
        6: "invalid_syntax",
    }
    masker = SimpleJsonGrammarMasker(vocab_id_to_text=vocab)

    # In initial state, invalid syntax should be masked
    initial_logits = np.zeros(len(vocab), dtype=np.float32)
    masked = masker.compute_mask(initial_logits)
    assert masked[0] == 0.0  # '{' is allowed
    assert np.isneginf(masked[6])  # 'invalid_syntax' is masked

    # Transition into object
    masker.update_state("{")
    assert masker.current_state == "IN_OBJECT_EXPECT_KEY_OR_CLOSE"
    assert masker.brace_depth == 1

    # Close object
    masker.update_state("}")
    assert masker.current_state == "COMPLETED"
    assert masker.brace_depth == 0

    # In completed state, non-empty tokens must be masked
    final_masked = masker.compute_mask(initial_logits)
    assert np.isneginf(final_masked[0])
    assert np.isneginf(final_masked[4])


def test_incremental_utf8_decoder_traditional_chinese_and_emoji() -> None:
    decoder = IncrementalUtf8Decoder()

    # Traditional Chinese word: '繁體' (繁 = 0xe7 0xb9 0x81, 體 = 0xe9 0xab 0x94)
    # Split byte sequence across 3 chunks deliberately
    full_bytes = "繁體中文😊".encode()

    chunk1 = full_bytes[:2]  # Incomplete first character
    chunk2 = full_bytes[2:5]  # Completes '繁' and starts '體'
    chunk3 = full_bytes[5:]  # Completes '體', '中', '文', and '😊'

    res1 = decoder.decode_incremental(chunk1)
    assert res1 == ""  # Cannot decode incomplete 2-byte prefix

    res2 = decoder.decode_incremental(chunk2)
    assert res2 == "繁"  # '繁' completed

    res3 = decoder.decode_incremental(chunk3)
    assert res3 == "體中文😊"  # Rest decoded cleanly without replacement errors


def test_edge_streaming_gateway() -> None:
    vocab_bytes = {
        101: "你好".encode(),
        102: "，世界".encode(),
        103: "！".encode(),
    }
    gateway = EdgeStreamingGateway(vocab_id_to_bytes=vocab_bytes)

    out1 = gateway.process_token(101)
    assert out1 == "你好"

    out2 = gateway.process_token(102)
    assert out2 == "，世界"

    out3 = gateway.process_token(103)
    assert out3 == "！"

    sse_frame = gateway.format_sse_chunk("你好", is_done=False)
    assert sse_frame == 'data: {"delta": "你好", "done": false}\n\n'
    assert tuple(gateway.token_history) == (101, 102, 103)
