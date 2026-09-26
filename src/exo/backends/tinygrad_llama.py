"""Llama decoder blocks evaluated on the active tinygrad device.

Qwen2 uses this block unchanged. Qwen3 adds query and key normalization when
those weights are present; that check lives in ``tinygrad_qwen`` so this
module does not import it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import final

from tinygrad import Tensor, dtypes

from exo.backends.tinygrad_weights import (
    RopeScalingConfig,
    TinygradModelSupportError,
    TinygradShardRoleError,
    TinygradWeightError,
    TransformerArchitecture,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


@final
@dataclass(frozen=True)
class AttentionParameters:
    query_projection: Tensor
    key_projection: Tensor
    value_projection: Tensor
    output_projection: Tensor
    query_bias: Tensor | None
    key_bias: Tensor | None
    value_bias: Tensor | None
    output_bias: Tensor | None
    query_norm_weight: Tensor | None
    key_norm_weight: Tensor | None


@final
@dataclass(frozen=True)
class MultilayerPerceptronParameters:
    gate_projection: Tensor
    up_projection: Tensor
    down_projection: Tensor


@final
@dataclass(frozen=True)
class DecoderLayerParameters:
    input_norm_weight: Tensor
    attention: AttentionParameters
    post_attention_norm_weight: Tensor
    multilayer_perceptron: MultilayerPerceptronParameters


@final
@dataclass(frozen=True)
class LoadedShard:
    """One pipeline rank's realized weights. The key-value cache is separate."""

    architecture: TransformerArchitecture
    layers: tuple[DecoderLayerParameters, ...]
    token_embedding: Tensor | None
    final_norm_weight: Tensor | None
    language_model_head: Tensor | None
    is_first_layer: bool
    is_last_layer: bool


@final
class LocalKeyValueCache:
    """On-device key and value cache for the layers owned by this rank.

    The cache is the mutable state of a forward pass. Hidden states that cross
    the network hop are not stored here.
    """

    def __init__(self, layer_count: int) -> None:
        self._keys: list[Tensor | None] = [None] * layer_count
        self._values: list[Tensor | None] = [None] * layer_count

    def cached(self, layer_offset: int) -> tuple[Tensor | None, Tensor | None]:
        return self._keys[layer_offset], self._values[layer_offset]

    def store(self, layer_offset: int, keys: Tensor, values: Tensor) -> None:
        self._keys[layer_offset] = keys
        self._values[layer_offset] = values


def _layer_candidates(layer_index: int, suffix: str) -> tuple[str, str]:
    return (
        f"model.layers.{layer_index}.{suffix}",
        f"layers.{layer_index}.{suffix}",
    )


def _optional_parameter(
    parameters: Mapping[str, Tensor], layer_index: int, suffix: str
) -> Tensor | None:
    for candidate in _layer_candidates(layer_index, suffix):
        found = parameters.get(candidate)
        if found is not None:
            return found
    return None


def _required_parameter(
    parameters: Mapping[str, Tensor], layer_index: int, suffix: str
) -> Tensor:
    found = _optional_parameter(parameters, layer_index, suffix)
    if found is None:
        raise TinygradWeightError(f"Layer {layer_index} is missing {suffix}")
    return found


def _named_parameter(
    parameters: Mapping[str, Tensor], *candidates: str
) -> Tensor | None:
    for candidate in candidates:
        found = parameters.get(candidate)
        if found is not None:
            return found
    return None


def require_qwen3_query_key_normalization(layer: DecoderLayerParameters) -> None:
    """Reject a Qwen3 layer that has no query or key normalization.

    Raises:
        TinygradModelSupportError: The runner entrypoint handles this by
            publishing ``RunnerTerminationError``.
    """
    if (
        layer.attention.query_norm_weight is None
        or layer.attention.key_norm_weight is None
    ):
        raise TinygradModelSupportError(
            "Qwen3 attention requires query and key normalization weights"
        )


def assemble_loaded_shard(
    parameters: Mapping[str, Tensor],
    architecture: TransformerArchitecture,
    shard: PipelineShardMetadata,
) -> LoadedShard:
    """Group realized tensors into decoder layers for ``shard``.

    Raises:
        TinygradWeightError: The runner entrypoint handles a missing projection,
            a missing final norm, or a missing language-model head.
        TinygradModelSupportError: The runner entrypoint handles a Qwen3 shard
            whose attention weights have no query or key normalization.
    """
    if architecture.num_attention_heads % architecture.resolved_key_value_heads() != 0:
        raise TinygradWeightError(
            "num_attention_heads must be divisible by num_key_value_heads"
        )
    layers: list[DecoderLayerParameters] = []
    for layer_index in range(shard.start_layer, shard.end_layer):
        query_bias = _optional_parameter(
            parameters, layer_index, "self_attn.q_proj.bias"
        )
        key_bias = _optional_parameter(parameters, layer_index, "self_attn.k_proj.bias")
        value_bias = _optional_parameter(
            parameters, layer_index, "self_attn.v_proj.bias"
        )
        output_bias = _optional_parameter(
            parameters, layer_index, "self_attn.o_proj.bias"
        )
        if architecture.attention_bias and (
            query_bias is None or key_bias is None or value_bias is None
        ):
            raise TinygradWeightError(
                f"Layer {layer_index} is missing attention bias required by config.json"
            )
        layer = DecoderLayerParameters(
            input_norm_weight=_required_parameter(
                parameters, layer_index, "input_layernorm.weight"
            ),
            attention=AttentionParameters(
                query_projection=_required_parameter(
                    parameters, layer_index, "self_attn.q_proj.weight"
                ),
                key_projection=_required_parameter(
                    parameters, layer_index, "self_attn.k_proj.weight"
                ),
                value_projection=_required_parameter(
                    parameters, layer_index, "self_attn.v_proj.weight"
                ),
                output_projection=_required_parameter(
                    parameters, layer_index, "self_attn.o_proj.weight"
                ),
                query_bias=query_bias,
                key_bias=key_bias,
                value_bias=value_bias,
                output_bias=output_bias,
                query_norm_weight=_optional_parameter(
                    parameters, layer_index, "self_attn.q_norm.weight"
                ),
                key_norm_weight=_optional_parameter(
                    parameters, layer_index, "self_attn.k_norm.weight"
                ),
            ),
            post_attention_norm_weight=_required_parameter(
                parameters, layer_index, "post_attention_layernorm.weight"
            ),
            multilayer_perceptron=MultilayerPerceptronParameters(
                gate_projection=_required_parameter(
                    parameters, layer_index, "mlp.gate_proj.weight"
                ),
                up_projection=_required_parameter(
                    parameters, layer_index, "mlp.up_proj.weight"
                ),
                down_projection=_required_parameter(
                    parameters, layer_index, "mlp.down_proj.weight"
                ),
            ),
        )
        if architecture.model_type == "qwen3":
            require_qwen3_query_key_normalization(layer)
        layers.append(layer)

    token_embedding = _named_parameter(
        parameters, "model.embed_tokens.weight", "embed_tokens.weight"
    )
    language_model_head = _named_parameter(parameters, "lm_head.weight")
    final_norm_weight = _named_parameter(parameters, "model.norm.weight")
    if shard.is_first_layer and token_embedding is None:
        raise TinygradWeightError("The first rank is missing embed_tokens.weight")
    if shard.is_last_layer and final_norm_weight is None:
        raise TinygradWeightError("The last rank is missing model.norm.weight")
    if shard.is_last_layer and language_model_head is None:
        if architecture.tie_word_embeddings and token_embedding is not None:
            language_model_head = token_embedding
        else:
            raise TinygradWeightError("The last rank is missing lm_head.weight")
    if not shard.is_first_layer:
        token_embedding = None
    if not shard.is_last_layer:
        final_norm_weight = None
        language_model_head = None
    return LoadedShard(
        architecture=architecture,
        layers=tuple(layers),
        token_embedding=token_embedding,
        final_norm_weight=final_norm_weight,
        language_model_head=language_model_head,
        is_first_layer=shard.is_first_layer,
        is_last_layer=shard.is_last_layer,
    )


def root_mean_square_normalize(
    hidden_state: Tensor, weight: Tensor, epsilon: float
) -> Tensor:
    variance = hidden_state.float().square().mean(axis=-1, keepdim=True)
    normalized = hidden_state.float() * (variance + epsilon).rsqrt()
    return (normalized * weight.float()).cast(hidden_state.dtype)


def _scale_inverse_frequencies(inverse: Tensor, scaling: RopeScalingConfig) -> Tensor:
    kind = scaling.scaling_kind()
    if kind == "none":
        return inverse
    if kind == "linear":
        return inverse / scaling.factor
    low_freq_factor = scaling.low_freq_factor
    high_freq_factor = scaling.high_freq_factor
    original_context = scaling.original_max_position_embeddings
    if low_freq_factor is None or high_freq_factor is None or original_context is None:
        raise TinygradModelSupportError(
            "Llama-3 rope_scaling requires low_freq_factor, high_freq_factor, "
            "and original_max_position_embeddings"
        )
    # Hugging Face Llama-3 scaling: long wavelengths are divided by ``factor``,
    # short wavelengths stay put, and the middle band is interpolated.
    low_wavelength = float(original_context) / low_freq_factor
    high_wavelength = float(original_context) / high_freq_factor
    wavelength = inverse.reciprocal() * (2.0 * math.pi)
    scaled = inverse / scaling.factor
    adjusted = (wavelength > low_wavelength).where(scaled, inverse)
    smooth = (wavelength.reciprocal() * float(original_context) - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (scaled * (smooth * -1.0 + 1.0)) + (inverse * smooth)
    medium = (wavelength <= low_wavelength) & (wavelength >= high_wavelength)
    return medium.where(smoothed, adjusted)


def rotary_inverse_frequencies(architecture: TransformerArchitecture) -> Tensor:
    head_dimension = architecture.resolved_head_dimension()
    indices = Tensor.arange(0, head_dimension, 2).float()
    exponent = indices / float(head_dimension)
    inverse = (
        Tensor(architecture.rope_theta, dtype=dtypes.float32).pow(exponent).reciprocal()
    )
    scaling = architecture.rope_scaling
    if scaling is None:
        return inverse
    return _scale_inverse_frequencies(inverse, scaling)


def _rotate_half(states: Tensor) -> Tensor:
    head_dimension = int(states.shape[-1])
    half = head_dimension // 2
    first = states.shrink((None, None, None, (0, half)))
    second = states.shrink((None, None, None, (half, head_dimension)))
    return (second * -1.0).cat(first, dim=-1)


def _apply_rotary_positions(
    query_states: Tensor,
    key_states: Tensor,
    architecture: TransformerArchitecture,
    past_length: int,
) -> tuple[Tensor, Tensor]:
    sequence_length = int(query_states.shape[2])
    inverse = rotary_inverse_frequencies(architecture)
    frequency_count = int(inverse.shape[0])
    positions = Tensor.arange(past_length, past_length + sequence_length).float()
    frequencies = positions.reshape(sequence_length, 1) * inverse.reshape(
        1, frequency_count
    )
    duplicated = frequencies.cat(frequencies, dim=-1)
    cosine = duplicated.cos().reshape(1, 1, sequence_length, int(duplicated.shape[1]))
    sine = duplicated.sin().reshape(1, 1, sequence_length, int(duplicated.shape[1]))
    rotated_query = (query_states.float() * cosine) + (
        _rotate_half(query_states.float()) * sine
    )
    rotated_key = (key_states.float() * cosine) + (
        _rotate_half(key_states.float()) * sine
    )
    return rotated_query.cast(query_states.dtype), rotated_key.cast(key_states.dtype)


def _project(hidden_state: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    projected = hidden_state.matmul(weight.transpose())
    if bias is None:
        return projected
    return projected + bias


def _split_heads(projected: Tensor, head_count: int, head_dimension: int) -> Tensor:
    batch = int(projected.shape[0])
    sequence = int(projected.shape[1])
    return projected.reshape(batch, sequence, head_count, head_dimension).permute(
        0, 2, 1, 3
    )


def _merge_heads(states: Tensor) -> Tensor:
    batch = int(states.shape[0])
    heads = int(states.shape[1])
    sequence = int(states.shape[2])
    head_dimension = int(states.shape[3])
    return states.permute(0, 2, 1, 3).reshape(batch, sequence, heads * head_dimension)


def _repeat_key_value_heads(states: Tensor, group_count: int) -> Tensor:
    if group_count == 1:
        return states
    batch = int(states.shape[0])
    key_value_heads = int(states.shape[1])
    sequence = int(states.shape[2])
    head_dimension = int(states.shape[3])
    return (
        states.reshape(batch, key_value_heads, 1, sequence, head_dimension)
        .expand(batch, key_value_heads, group_count, sequence, head_dimension)
        .reshape(batch, key_value_heads * group_count, sequence, head_dimension)
    )


def _masked_attention_scores(scores: Tensor, past_length: int) -> Tensor:
    query_length = int(scores.shape[2])
    key_length = int(scores.shape[3])
    query_index = Tensor.arange(past_length, past_length + query_length).reshape(
        1, 1, query_length, 1
    )
    key_index = Tensor.arange(0, key_length).reshape(1, 1, 1, key_length)
    allowed = key_index <= query_index
    negative = Tensor.full(
        (
            int(scores.shape[0]),
            int(scores.shape[1]),
            query_length,
            key_length,
        ),
        -1.0e4,
        dtype=dtypes.float32,
    )
    return allowed.where(scores.float(), negative)


def _swiglu(hidden_state: Tensor, multilayer: MultilayerPerceptronParameters) -> Tensor:
    gate = _project(hidden_state, multilayer.gate_projection, None).silu()
    up_projection = _project(hidden_state, multilayer.up_projection, None)
    return _project(gate * up_projection, multilayer.down_projection, None)


def _attention(
    hidden_state: Tensor,
    layer: DecoderLayerParameters,
    architecture: TransformerArchitecture,
    cached_keys: Tensor | None,
    cached_values: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    attention = layer.attention
    head_dimension = architecture.resolved_head_dimension()
    query_head_count = architecture.num_attention_heads
    key_value_head_count = architecture.resolved_key_value_heads()
    group_count = query_head_count // key_value_head_count
    activation_dtype = hidden_state.dtype
    query_states = _split_heads(
        _project(hidden_state, attention.query_projection, attention.query_bias),
        query_head_count,
        head_dimension,
    )
    key_states = _split_heads(
        _project(hidden_state, attention.key_projection, attention.key_bias),
        key_value_head_count,
        head_dimension,
    )
    value_states = _split_heads(
        _project(hidden_state, attention.value_projection, attention.value_bias),
        key_value_head_count,
        head_dimension,
    )
    if attention.query_norm_weight is not None:
        query_states = root_mean_square_normalize(
            query_states, attention.query_norm_weight, architecture.rms_norm_eps
        )
    if attention.key_norm_weight is not None:
        key_states = root_mean_square_normalize(
            key_states, attention.key_norm_weight, architecture.rms_norm_eps
        )
    past_length = 0 if cached_keys is None else int(cached_keys.shape[2])
    query_states, key_states = _apply_rotary_positions(
        query_states, key_states, architecture, past_length
    )
    if cached_keys is not None and cached_values is not None:
        key_states = cached_keys.cat(key_states, dim=2)
        value_states = cached_values.cat(value_states, dim=2)
    updated_keys = key_states.contiguous().realize()
    updated_values = value_states.contiguous().realize()
    repeated_keys = _repeat_key_value_heads(updated_keys, group_count)
    repeated_values = _repeat_key_value_heads(updated_values, group_count)
    scale = 1.0 / math.sqrt(float(head_dimension))
    scores = (
        query_states.float().matmul(repeated_keys.float().transpose(-2, -1)) * scale
    )
    masked = _masked_attention_scores(scores, past_length)
    weights = masked.softmax(axis=-1)
    attended = weights.matmul(repeated_values.float())
    merged = _merge_heads(attended).cast(activation_dtype)
    output = _project(merged, attention.output_projection, attention.output_bias)
    return output.cast(activation_dtype), updated_keys, updated_values


def forward_llama_decoder_layer(
    hidden_state: Tensor,
    layer: DecoderLayerParameters,
    architecture: TransformerArchitecture,
    cached_keys: Tensor | None,
    cached_values: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one Llama block and return the realized hidden state plus cache.

    The returned key and value tensors include ``cached_keys`` and
    ``cached_values`` when those arguments are present. Query positions start
    at the cached length.
    """
    activation_dtype = hidden_state.dtype
    normalized = root_mean_square_normalize(
        hidden_state, layer.input_norm_weight, architecture.rms_norm_eps
    )
    attention_output, keys, values = _attention(
        normalized, layer, architecture, cached_keys, cached_values
    )
    residual = hidden_state.float() + attention_output.float()
    multilayer_input = root_mean_square_normalize(
        residual.cast(activation_dtype),
        layer.post_attention_norm_weight,
        architecture.rms_norm_eps,
    )
    multilayer_output = _swiglu(multilayer_input, layer.multilayer_perceptron)
    updated = (
        (residual + multilayer_output.float())
        .cast(activation_dtype)
        .contiguous()
        .realize()
    )
    return updated, keys, values


def forward_shard_layer(
    hidden_state: Tensor,
    layer: DecoderLayerParameters,
    architecture: TransformerArchitecture,
    cached_keys: Tensor | None,
    cached_values: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dispatch one layer to the Llama, Qwen2, or Qwen3 block."""
    if architecture.model_type == "qwen3":
        from exo.backends.tinygrad_qwen import forward_qwen3_decoder_layer

        return forward_qwen3_decoder_layer(
            hidden_state, layer, architecture, cached_keys, cached_values
        )
    if architecture.model_type == "qwen2":
        from exo.backends.tinygrad_qwen import forward_qwen2_decoder_layer

        return forward_qwen2_decoder_layer(
            hidden_state, layer, architecture, cached_keys, cached_values
        )
    return forward_llama_decoder_layer(
        hidden_state, layer, architecture, cached_keys, cached_values
    )


def embed_token_tensor(loaded_shard: LoadedShard, token_ids: Tensor) -> Tensor:
    """Look up token rows on the first rank.

    Raises:
        TinygradShardRoleError: The runner entrypoint handles this when a
            later rank requests embeddings.
        TinygradWeightError: The runner entrypoint handles this when the first
            rank has no embedding table.
    """
    if not loaded_shard.is_first_layer:
        raise TinygradShardRoleError(
            "embed_token_ids is only valid on the first pipeline rank"
        )
    token_embedding = loaded_shard.token_embedding
    if token_embedding is None:
        raise TinygradWeightError("The first rank has no embed_tokens.weight")
    return token_embedding[token_ids].contiguous().realize()


def project_logits_tensor(loaded_shard: LoadedShard, hidden_state: Tensor) -> Tensor:
    """Apply the final norm and language-model head on the last rank.

    Raises:
        TinygradShardRoleError: The runner entrypoint handles this when an
            earlier rank requests logits.
        TinygradWeightError: The runner entrypoint handles a last rank that is
            missing the final norm or the language-model head.
    """
    if not loaded_shard.is_last_layer:
        raise TinygradShardRoleError(
            "project_logits is only valid on the last pipeline rank"
        )
    final_norm_weight = loaded_shard.final_norm_weight
    language_model_head = loaded_shard.language_model_head
    if final_norm_weight is None or language_model_head is None:
        raise TinygradWeightError("The last rank cannot project logits")
    normalized = root_mean_square_normalize(
        hidden_state, final_norm_weight, loaded_shard.architecture.rms_norm_eps
    )
    return normalized.matmul(language_model_head.transpose()).contiguous().realize()


def forward_loaded_shard(
    loaded_shard: LoadedShard,
    cache: LocalKeyValueCache,
    hidden_state: Tensor,
) -> Tensor:
    """Run every layer assigned to this rank, appending to ``cache``."""
    tensor = hidden_state
    for layer_offset, layer in enumerate(loaded_shard.layers):
        cached_keys, cached_values = cache.cached(layer_offset)
        tensor, keys, values = forward_shard_layer(
            tensor,
            layer,
            loaded_shard.architecture,
            cached_keys,
            cached_values,
        )
        cache.store(layer_offset, keys, values)
    return tensor
