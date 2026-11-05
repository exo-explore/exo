"""The stub engine on its own: deterministic replies, rank agreement, and losing a peer."""

import re
import socket
import threading
import time
from pathlib import Path

import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.testkit.cluster import StubModel
from exo.testkit.stub import ring as ring_module
from exo.testkit.stub.config import STUB_CONFIG_ENV, StubConfig, write_config
from exo.testkit.stub.directives import (
    Directives,
    expected_reply,
    parse_directives,
)
from exo.testkit.stub.engine import StubBuilder, StubEngine
from exo.testkit.stub.ring import LOST_PEER_MESSAGE, Ring, RingError
from exo.utils.channels import MpReceiver, mp_channel

MODEL = ModelId("exo-testkit/stub-8l")


def _task(
    prompt: str, max_tokens: int | None = None, name: str = "a"
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(f"task-{name}"),
        instance_id=InstanceId("instance"),
        command_id=CommandId(f"command-{name}"),
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent(prompt))],
            stream=True,
            max_output_tokens=max_tokens,
        ),
    )


def _cancels() -> MpReceiver[TaskId]:
    _, receiver = mp_channel[TaskId]()
    return receiver


def _run(engine: StubEngine, task: TextGeneration, steps: int = 10_000) -> list[object]:
    engine.submit(task)
    results: list[object] = []
    for _ in range(steps):
        for task_id, result in engine.step():
            assert task_id == task.task_id
            results.append(result)
            if isinstance(result, (FinishedResponse, CancelledResponse)):
                return results
    raise AssertionError("the engine never finished the request")


def test_directives_round_trip_through_their_tag() -> None:
    directives = Directives(
        tokens=40, rate=20, fault="stall", fault_at=3, fault_rank=1, stall_seconds=2
    )
    assert parse_directives(f"hello {directives.tag()} world") == directives
    assert parse_directives("no tag here") == Directives()


@pytest.mark.parametrize(
    "prompt", ["[stub tokns=3]", "[stub fault=explode]", "[stub tokens]"]
)
def test_a_directive_the_stub_does_not_know_is_an_error(prompt: str) -> None:
    with pytest.raises(ValueError):
        parse_directives(prompt)


def test_the_reply_is_a_function_of_the_prompt() -> None:
    assert expected_reply("p [stub tokens=3]") == expected_reply("p [stub tokens=3]")
    assert (
        expected_reply("p [stub tokens=3]").text
        != expected_reply("q [stub tokens=3]").text
    )
    short = expected_reply("p [stub tokens=5]", max_tokens=2)
    assert len(short.tokens) == 2
    assert short.finish_reason == "length"


def test_one_rank_generates_exactly_the_expected_tokens() -> None:
    prompt = "count [stub tokens=7 rate=1000]"
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    results = _run(engine, _task(prompt))
    chunks = [r for r in results if isinstance(r, TokenChunk)]
    assert "".join(chunk.text for chunk in chunks) == expected_reply(prompt).text
    assert [chunk.finish_reason for chunk in chunks] == [None] * 6 + ["stop"]
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 7
    assert isinstance(results[-1], FinishedResponse)


def test_the_rate_holds_when_sleeps_run_late(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under load macOS stretches short sleeps (a 20 ms sleep took 127 ms on a busy Mac): the
    stub makes up for late steps instead of falling behind."""
    real_sleep = time.sleep

    def late_sleep(seconds: float) -> None:
        real_sleep(seconds + 0.1)

    monkeypatch.setattr(time, "sleep", late_sleep)
    prompt = "steady [stub tokens=50 rate=50]"
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    started = time.monotonic()
    results = _run(engine, _task(prompt))
    took = time.monotonic() - started
    assert "".join(r.text for r in results if isinstance(r, TokenChunk)) == (
        expected_reply(prompt).text
    )
    # 50 tokens at 50 a second: the last is due after 0.98 s
    assert 0.9 < took < 1.3, f"{took:.2f}s"


def test_prefill_delays_the_first_token() -> None:
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    started = time.monotonic()
    _run(engine, _task("think [stub tokens=2 rate=1000 prefill=0.3]"))
    took = time.monotonic() - started
    assert 0.3 <= took < 1.0, f"{took:.2f}s"


def test_load_and_warmup_take_their_time_and_meet_their_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-node stub config: each phase sleeps its time, and a fault set for a phase
    happens there (read at the start of each phase, so a test can change it between)."""
    path = tmp_path / "stub.json"
    monkeypatch.setenv(STUB_CONFIG_ENV, str(path))
    write_config(path, StubConfig(load_seconds=0.2, warmup_seconds=0.1))
    (bound,) = _ring_instance(1)
    sender, _ = mp_channel[Event]()
    builder = StubBuilder(MODEL, sender, _cancels())

    started = time.monotonic()
    progress = list(builder.load(bound))
    assert [response.layers_loaded for response in progress] == list(range(1, 9))
    assert time.monotonic() - started >= 0.2
    engine = builder.build()
    started = time.monotonic()
    engine.warmup()
    assert time.monotonic() - started >= 0.1

    write_config(path, StubConfig(fault_phase="warmup", fault="raise"))
    with pytest.raises(RuntimeError, match="raise in warmup"):
        engine.warmup()


def test_max_tokens_cuts_the_reply_short() -> None:
    prompt = "count [stub tokens=7 rate=1000]"
    engine = StubEngine(MODEL, 0, 1, None, _cancels())
    results = _run(engine, _task(prompt, max_tokens=3))
    text = "".join(r.text for r in results if isinstance(r, TokenChunk))
    assert text == expected_reply(prompt, max_tokens=3).text
    last = [r for r in results if isinstance(r, TokenChunk)][-1]
    assert last.finish_reason == "length"


def test_a_cancelled_request_ends_with_a_cancellation() -> None:
    sender, receiver = mp_channel[TaskId]()
    engine = StubEngine(MODEL, 0, 1, None, receiver)
    task = _task("long [stub tokens=1000 rate=1000]")
    engine.submit(task)
    for _ in range(3):
        list(engine.step())
    sender.send(task.task_id)
    results = [result for _ in range(5) for _, result in engine.step()]
    assert any(isinstance(result, CancelledResponse) for result in results)


def _free_ports_in_a_row(count: int) -> int:
    for _ in range(50):
        sockets = [socket.socket() for _ in range(count)]
        try:
            sockets[0].bind(("0.0.0.0", 0))
            port = int(sockets[0].getsockname()[1])  # pyright: ignore[reportAny]
            for offset, each in enumerate(sockets[1:], start=1):
                each.bind(("0.0.0.0", port + offset))
            return port
        except OSError:
            continue
        finally:
            for each in sockets:
                each.close()
    raise AssertionError(f"no {count} free ports in a row")


def _ring_instance(size: int) -> list[BoundInstance]:
    port = _free_ports_in_a_row(size)
    nodes = [NodeId(f"node-{rank}") for rank in range(size)]
    runners = [RunnerId(f"runner-{rank}") for rank in range(size)]
    card = StubModel().card()
    layers = 8 // size
    instance = MlxRingInstance(
        instance_id=InstanceId("instance"),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runners[rank]: PipelineShardMetadata(
                    model_card=card,
                    device_rank=rank,
                    world_size=size,
                    start_layer=layers * rank,
                    end_layer=layers * rank + layers,
                    n_layers=8,
                )
                for rank in range(size)
            },
            node_to_runner=dict(zip(nodes, runners, strict=True)),
        ),
        hosts_by_node={
            node: [
                Host(ip="0.0.0.0" if other == rank else "127.0.0.1", port=port)
                for other in range(size)
            ]
            for rank, node in enumerate(nodes)
        },
        ephemeral_port=port,
    )
    return [
        BoundInstance(
            instance=instance, bound_runner_id=runners[rank], bound_node_id=nodes[rank]
        )
        for rank in range(size)
    ]


def _connect_all(size: int) -> list[Ring]:
    bound = _ring_instance(size)
    rings: dict[int, Ring] = {}

    def join(rank: int) -> None:
        rings[rank] = Ring.connect(bound[rank], proxy_port=None)

    # As exo's workers do: every rank but the last first, then the last
    threads = [threading.Thread(target=join, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert set(rings) == set(range(size)), "the ranks didn't connect"
    return [rings[rank] for rank in range(size)]


def _connect_both() -> tuple[Ring, Ring]:
    first, second = _connect_all(2)
    return first, second


def test_two_ranks_agree_and_only_rank_zero_reports_tokens() -> None:
    rings = _connect_both()
    prompt = "together [stub tokens=12 rate=1000]"
    results: dict[int, list[object]] = {}

    def run(rank: int) -> None:
        engine = StubEngine(MODEL, rank, 2, rings[rank], _cancels())
        results[rank] = _run(engine, _task(prompt))

    threads = [threading.Thread(target=run, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert set(results) == {0, 1}
    text = "".join(r.text for r in results[0] if isinstance(r, TokenChunk))
    assert text == expected_reply(prompt).text
    assert not any(isinstance(r, TokenChunk) for r in results[1])
    assert isinstance(results[0][-1], FinishedResponse)
    assert isinstance(results[1][-1], FinishedResponse)
    for ring in rings:
        ring.close()


def test_a_rank_that_loses_its_peer_fails_as_mlx_does() -> None:
    first, second = _connect_both()
    second.close()
    with pytest.raises(RingError, match=re.escape(LOST_PEER_MESSAGE)):
        first.all_gather(b"anyone there?")
    first.close()


def test_three_ranks_gather_every_rank_s_payload() -> None:
    rings = _connect_all(3)
    gathered: dict[int, list[bytes]] = {}

    def gather(rank: int) -> None:
        gathered[rank] = rings[rank].all_gather(f"from {rank}".encode())

    threads = [threading.Thread(target=gather, args=(rank,)) for rank in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    expected = [f"from {rank}".encode() for rank in range(3)]
    assert gathered == {rank: expected for rank in range(3)}
    for ring in rings:
        ring.close()


def test_the_connecting_rank_gives_up_on_mlx_s_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 1 of 2 connects; with nobody listening it tries 5 times, then fails."""
    monkeypatch.setattr(ring_module, "DIAL_BACKOFF_SECONDS", (0.01,) * 5)
    _, last = _ring_instance(2)
    with pytest.raises(RuntimeError, match="Couldn't connect"):
        Ring.connect(last, proxy_port=None)


def test_a_second_runner_on_the_port_binds_and_waits() -> None:
    """MLX listens with SO_REUSEPORT: a new runner can bind while an old one still listens."""
    first, _ = _ring_instance(2)
    port = (
        first.instance.ephemeral_port
        if isinstance(first.instance, MlxRingInstance)
        else 0
    )
    old = socket.socket()
    old.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    old.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    old.bind(("127.0.0.1", port))
    old.listen()
    joined: list[Ring] = []
    waiting = threading.Thread(
        target=lambda: joined.append(Ring.connect(first, proxy_port=None)), daemon=True
    )
    waiting.start()
    waiting.join(0.5)
    assert waiting.is_alive(), "rank 0 should be waiting in accept, not failing to bind"
    old.close()
    # Let it finish: whatever connects and says it is rank 1 pairs with it
    with socket.create_connection(("127.0.0.1", port)) as peer:
        peer.sendall(b"1\n")
        waiting.join(5)
        assert joined, "rank 0 never accepted"
        joined[0].close()
