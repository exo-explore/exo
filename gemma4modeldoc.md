# Gemma4

## Overview

Gemma 4 is a multimodal model with pretrained and instruction-tuned variants, available in E2B, E4B, 31B and 26B-A4B (MoE) parameter sizes. Gemma 4 models provide the following capabilities:
- Reasoning: All models in the family are designed as highly capable reasoners, with configurable thinking modes.
- Extended Multimodalities: Processes Text, Image with variable aspect ratio and resolution support (all models), Video, and Audio (featured natively on the E2B and E4B models).
- Increased Context Window: Small models feature a 128K context window, while the other models support 256K.
- Enhanced Coding & Agentic Capabilities: Achieves notable improvements in coding benchmarks alongside built-in function-calling support, powering highly capable autonomous agents.
- Native System Prompt Support: Gemma 4 introduces built-in support for the system role, enabling more structured and controllable conversations.

You can find all the original Gemma 4 checkpoints under the [Gemma 4](https://huggingface.co/collections/google/gemma-4) release.

### Gemma4 Vision Model

The key difference from previous Gemma releases for vision is the new design to process **images of different sizes** using a **fixed-budget number of tokens**. Unlike many models that squash every image into a fixed square (like 224×224), Gemma 4 keeps the image's natural aspect ratio while making it the right size. There are a couple constraints to follow:
- The total number of pixels must fit within a patch budget
- Both height and width must be divisible by **48** (= patch size 16 × pooling kernel 3)

> [!IMPORTANT]
> Gemma 4 does **not** apply the standard ImageNet mean/std normalization that many other vision models use. The model's own patch embedding layer handles the final scaling internally (shifting values to the [-1, 1] range).

The number of "soft tokens" (aka vision tokens) an image processor can produce is configurable. The supported options are outlined below and the default is **280 soft tokens** per image.

| Soft Tokens | Patches (before pooling) | Approx. Image Area |
|:-----------:|:------------------------:|:-------------------:|
| 70          | 630                      | ~161K pixels        |
| 140         | 1,260                    | ~323K pixels        |
| **280**     | **2,520**                | **~645K pixels**    |
| 560         | 5,040                    | ~1.3M pixels        |
| 1,120       | 10,080                   | ~2.6M pixels        |

To encode positional information for each patch in the image, Gemma 4 uses a learned 2D position embedding table. The position table stores up to 10,240 positions per axis, which allows the model to handle very large images. Each position is a learned vector of the same dimensions as the patch embedding. The 2D RoPE which Gemma 4 uses independently rotate half the attention head dimensions for the x-axis and the other half for the y-axis. This allows the model to understand spatial relationships like "above," "below," "left of," and "right of."

### Per-Layer Embeddings (PLE)

Gemma 4 introduces a Per-Layer Embeddings (PLE) system that feeds an auxiliary residual signal into each decoder layer, rather than relying solely on a single shared embedding at the input.

PLE combines two components that are summed and scaled by `1/√2` before being fed to each decoder layer:

1. Token-identity (`get_per_layer_inputs`): looks up `input_ids` in `embed_tokens_per_layer`, a `Gemma4TextScaledWordEmbedding` that multiplies by `√(hidden_size_per_layer_input)`. The packed output is reshaped from `[batch, seq, num_hidden_layers * hidden_size_per_layer_input]` to `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.
2. Context-aware (`project_per_layer_inputs`): projects `inputs_embeds` through `per_layer_model_projection` (a Linear layer), scales by `1/√(hidden_size)`, reshapes to `[batch, seq, num_layers, ple_dim]`, and normalizes with `per_layer_projection_norm` (RMSNorm).

When both components are available, the final per-layer input is `(token_identity + context_aware) * (1/√2)`. For multimodal inputs where `input_ids` are not available, only the context-aware projection is used.

## Usage examples

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/gemma-4-E2B-it",
)
pipeline(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<|image|>\n\nWhat is shown in this image?"
)
```

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it",
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left"
)

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
```

### Function calling

```python
from transformers import AutoModelForCausalLM, AutoProcessor

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_n_day_weather_forecast",
        "description": "Get an N-day weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use",
                },
                "num_days": {
                    "type": "integer",
                    "description": "The number of days to forecast",
                },
            },
            "required": ["location", "format", "num_days"],
        },
    },
}

messages = [
    {
        "role": "user",
        "content": "What's the weather like the next 3 days in San Francisco, CA (using F)?",
    },
]

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left"
)

text = processor.apply_chat_template(
    messages,
    tools=[WEATHER_TOOL],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

### Audio (E2B and E4B Only)

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please transcribe the following audio:"},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
        ],
    }
]

model = AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left"
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)

input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

## Gemma4AudioConfig[[transformers.Gemma4AudioConfig]]

- **hidden_size** (`int`, *optional*, defaults to `1024`) --
  Dimension of the hidden representations.
- **num_hidden_layers** (`int`, *optional*, defaults to `12`) --
  Number of hidden layers in the Transformer decoder.
- **num_attention_heads** (`int`, *optional*, defaults to `8`) --
  Number of attention heads for each attention layer in the Transformer decoder.
- **hidden_act** (`str`, *optional*, defaults to `silu`) --
  The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
  `"relu"`, `"silu"`, etc.
- **subsampling_conv_channels** (`list[int]`, defaults to `[128, 32]`) --
  Channel sizes for the convolutional layers in the Sub-sample Convolution Projection.
- **conv_kernel_size** (`int`, *optional*, defaults to `5`) --
  The size of the convolutional kernel.
- **residual_weight** (`float`, defaults to `0.5`) --
  Scaling applied to hidden_states prior to combining with the residual in the feedforward.
- **attention_chunk_size** (`int`, defaults to `12`) --
  The sub-sequence size for attention processing.
- **attention_context_left** (`int`, defaults to `13`) --
  The leftward context size for the attention chunk.
- **attention_context_right** (`int`, defaults to `0`) --
  The rightward context size for the attention chunk.
- **attention_logit_cap** (`float`, defaults to `50.0`) --
  Cap applied to attention weights.
- **attention_invalid_logits_value** (`float`, defaults to `1e-9`) --
  Value to use for invalid logits in attention.
- **use_clipped_linears** (`bool`, defaults to `True`) --
  If true, apply clipping to the Linear layers, drawing bounds from the model checkpoint.
- **rms_norm_eps** (`float`, *optional*, defaults to `1e-06`) --
  The epsilon used by the rms normalization layers.
- **gradient_clipping** (`float`, defaults to `1e10`) --
  Clipping value used to stabilize extremely large gradient values.
- **output_proj_dims** (`int`, defaults to `1536`) --
  Dimension of the final linear projection from `hidden_size` to the model's output.
- **initializer_range** (`float`, *optional*, defaults to `0.02`) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Gemma4Model. It is used to instantiate a Gemma4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Gemma4VisionConfig[[transformers.Gemma4VisionConfig]]

- **hidden_size** (`int`, *optional*, defaults to `768`) --
  Dimension of the hidden representations.
- **intermediate_size** (`int`, *optional*, defaults to `3072`) --
  Dimension of the MLP representations.
- **num_hidden_layers** (`int`, *optional*, defaults to `16`) --
  Number of hidden layers in the Transformer decoder.
- **num_attention_heads** (`int`, *optional*, defaults to `12`) --
  Number of attention heads for each attention layer in the Transformer decoder.
- **num_key_value_heads** (`int`, *optional*, defaults to `12`) --
  This is the number of key_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
- **head_dim** (`int`, *optional*, defaults to `64`) --
  The attention head dimension. If None, it will default to hidden_size // num_attention_heads
- **hidden_activation** (`str`, *optional*, defaults to `gelu_pytorch_tanh`) --
  The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
  `"relu"`, `"silu"`, etc.
- **rms_norm_eps** (`float`, *optional*, defaults to `1e-06`) --
  The epsilon used by the rms normalization layers.
- **max_position_embeddings** (`int`, *optional*, defaults to `131072`) --
  The maximum sequence length that this model might ever be used with.
- **attention_bias** (`bool`, *optional*, defaults to `False`) --
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
- **attention_dropout** (`float`, *optional*, defaults to `0.0`) --
  The dropout ratio for the attention probabilities.
- **rope_parameters** (`dict`, *optional*) --
  Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
  a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
  with longer `max_position_embeddings`.
- **pooling_kernel_size** (`int`, *optional*) --
  Spatial pooling kernel size applied after patchification.
- **patch_size** (`int`, *optional*, defaults to `16`) --
  The size (resolution) of each patch.
- **position_embedding_size** (`int`, defaults to 10240) --
  Maximum number of position embeddings for the vision encoder. Controls the size of
  the learned 2D position embedding table used by the patch embedder.
- **use_clipped_linears** (`bool`, defaults to `False`) --
  Whether to use weight-clipped linear layers. When enabled, linear layer weights are
  clamped to a fixed range during the forward pass to improve numerical stability.
- **standardize** (`bool`, defaults to `False`) --
  If true, applies a bias and scale to the soft tokens returned from the pooler.
- **initializer_range** (`float`, *optional*, defaults to `0.02`) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Gemma4Model. It is used to instantiate a Gemma4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Gemma4TextConfig[[transformers.Gemma4TextConfig]]

- **vocab_size** (`int`, *optional*, defaults to `262144`) --
  Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.
- **hidden_size** (`int`, *optional*, defaults to `2304`) --
  Dimension of the hidden representations.
- **intermediate_size** (`int`, *optional*, defaults to `9216`) --
  Dimension of the MLP representations.
- **num_hidden_layers** (`int`, *optional*, defaults to `30`) --
  Number of hidden layers in the Transformer decoder.
- **num_attention_heads** (`int`, *optional*, defaults to `8`) --
  Number of attention heads for each attention layer in the Transformer decoder.
- **num_key_value_heads** (`int`, *optional*, defaults to `4`) --
  This is the number of key_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
- **head_dim** (`int`, *optional*, defaults to `256`) --
  The attention head dimension. If None, it will default to hidden_size // num_attention_heads
- **hidden_activation** (`str`, *optional*, defaults to `gelu_pytorch_tanh`) --
  The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
  `"relu"`, `"silu"`, etc.
- **max_position_embeddings** (`int`, *optional*, defaults to `131072`) --
  The maximum sequence length that this model might ever be used with.
- **initializer_range** (`float`, *optional*, defaults to `0.02`) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
- **rms_norm_eps** (`float`, *optional*, defaults to `1e-06`) --
  The epsilon used by the rms normalization layers.
- **use_cache** (`bool`, *optional*, defaults to `True`) --
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.
- **pad_token_id** (`int`, *optional*, defaults to `0`) --
  Token id used for padding in the vocabulary.
- **eos_token_id** (`Union[int, list[int]]`, *optional*, defaults to `1`) --
  Token id used for end-of-stream in the vocabulary.
- **bos_token_id** (`int`, *optional*, defaults to `2`) --
  Token id used for beginning-of-stream in the vocabulary.
- **tie_word_embeddings** (`bool`, *optional*, defaults to `True`) --
  Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.
- **rope_parameters** (`dict`, *optional*) --
  Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
  a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
  with longer `max_position_embeddings`.
- **attention_bias** (`bool`, *optional*, defaults to `False`) --
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
- **attention_dropout** (`Union[int, float]`, *optional*, defaults to `0.0`) --
  The dropout ratio for the attention probabilities.
- **sliding_window** (`int`, *optional*, defaults to `512`) --
  Sliding window attention window size. If `None`, no sliding window is applied.
- **layer_types** (`list[str]`, *optional*) --
  A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically
  generated based on config values.
- **final_logit_softcapping** (`float`, *optional*) --
  Soft-capping value applied to the final logits before computing the probability distribution. Logits are
  scaled by `tanh(logit / cap) * cap`.
- **use_bidirectional_attention** (`str`, *optional*) --
  Controls bidirectional attention behavior. When set to `"vision"`, vision tokens
  attend bidirectionally while text tokens use causal attention. When set to `"all"`,
  all tokens use bidirectional attention.
- **vocab_size_per_layer_input** (`int`, defaults to 262144) --
  Vocabulary size for the per-layer input embeddings (PLE). Used by models with
  per-layer residual streams where a smaller embedding is added at each decoder layer.
- **hidden_size_per_layer_input** (`int`, defaults to 256) --
  Per-layer hidden dimension for the PLE system. The actual embedding weight has shape
  `[vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]`
  because all layers are packed into a single table. See the [Gemma4](https://huggingface.co/docs/transformers/main/en/model_doc/gemma4#per-layer-embeddings-ple) docs
  for a description of the full PLE pipeline.
- **num_global_key_value_heads** (`int`, *optional*) --
  Number of key-value heads for global (full) attention layers. If `None`, defaults
  to `num_key_value_heads`.
- **global_head_dim** (`int`, defaults to 512) --
  Dimension of each attention head in global (full) attention layers.
- **attention_k_eq_v** (`bool`, defaults to `False`) --
  Whether keys and values share the same projection weights. When `True`, the key
  projection output is reused as the value projection.
- **num_kv_shared_layers** (`int`, defaults to 0) --
  Number of consecutive decoder layers that share the same key-value projections.
  A value of 0 means no sharing (each layer has independent KV projections).
- **enable_moe_block** (`bool`, defaults to `False`) --
  Whether to enable Mixture-of-Experts (MoE) blocks in the decoder layers. When
  `True`, eligible layers will use a sparse MoE feed-forward network.
- **use_double_wide_mlp** (`bool`, defaults to `False`) --
  Whether to use a double-width MLP with fused gate and up projections.
- **num_experts** (`int`, *optional*) --
  Number of routed experts in MoE layers.

- **top_k_experts** (`int`, *optional*) --
  Number of experts activated per token in MoE layers. Only used when
  `enable_moe_block=True`.
- **moe_intermediate_size** (`int`, *optional*) --
  Intermediate (hidden) size of each expert's feed-forward network in MoE layers.
  Only used when `enable_moe_block=True`.

This is the configuration class to store the configuration of a Gemma4Model. It is used to instantiate a Gemma4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Gemma4Config[[transformers.Gemma4Config]]

- **text_config** (`Union[~models.gemma4.configuration_gemma4.Gemma4TextConfig, dict[str, Any]]`, *optional*) --
  The config object or dictionary of the text backbone.
- **vision_config** (`Union[~models.gemma4.configuration_gemma4.Gemma4VisionConfig, dict[str, Any]]`, *optional*) --
  The config object or dictionary of the vision backbone.
- **audio_config** (`Union[~models.gemma4.configuration_gemma4.Gemma4AudioConfig, dict[str, Any]]`, *optional*) --
  The config object or dictionary of the audio backbone.
- **boi_token_id** (`int`, *optional*, defaults to 255999) --
  The begin-of-image token index to wrap the image prompt.
- **eoi_token_id** (`int`, *optional*, defaults to 258882) --
  The end-of-image token index to wrap the image prompt.
- **image_token_id** (`int`, *optional*, defaults to `258880`) --
  The image token index used as a placeholder for input images.
- **video_token_id** (`int`, *optional*, defaults to `258884`) --
  The video token index used as a placeholder for input videos.
- **boa_token_id** (`int`, *optional*, defaults to 256000) --
  The begin-of-audio token index to wrap the audio prompt.
- **eoa_token_index** (`int`, *optional*, defaults to 258883) --
  The end-of-audio token index to wrap the audio prompt.
- **audio_token_id** (`int`, *optional*, defaults to `258881`) --
  The audio token index used as a placeholder for input audio.
- **initializer_range** (`float`, *optional*, defaults to `0.02`) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
- **tie_word_embeddings** (`bool`, *optional*, defaults to `True`) --
  Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Gemma4Model. It is used to instantiate a Gemma4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import (
>>>     Gemma4AudioConfig,
>>>     Gemma4Config,
>>>     Gemma4ForConditionalGeneration,
>>>     Gemma4TextConfig,
>>>     Gemma4VisionConfig,
>>> )

>>> # Initializing a Gemma 4 Audio config.
>>> audio_config = Gemma4AudioConfig()

>>> # Initializing a Gemma 4 Text config.
>>> text_config = Gemma4TextConfig()

>>> # Initializing a Gemma 4 vision config.
>>> vision_config = Gemma4VisionConfig()

>>> # Initializing a Gemma 4 config similar to google/gemma-4-e2b-it
>>> configuration = Gemma4Config(text_config, vision_config, audio_config)

>>> # Initializing a model from the google/gemma-4-e2b-it configuration
>>> model = Gemma4ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Gemma4AudioFeatureExtractor[[transformers.Gemma4AudioFeatureExtractor]]

- **feature_size** (`int`, *optional*, defaults to 128) --
  The feature dimension of the extracted features.
- **sampling_rate** (`int`, *optional*, defaults to 16000) --
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
- **padding_value** (`float`, *optional*, defaults to 0.0) --
  Padding value used to pad the audio. Should correspond to silences.
- **return_attention_mask** (`bool`, *optional*, defaults to `True`) --
  Whether to return the attention mask for the generated MEL spectrograms.
- **frame_length_ms** (`float`, *optional*, defaults to 20.0) --
  The length of a frame in milliseconds.
- **hop_length_ms** (`float`, *optional*, defaults to 10.0) --
  Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
- **min_frequency** (`float`, *optional*, defaults to 0.0) --
  The minimum frequency (in Hz) for the Mel filterbank.
- **max_frequency** (`float`, *optional*, defaults to 8000.0) --
  The maximum frequency (in Hz) for the Mel filterbank.
- **preemphasis** (`float`, *optional*, defaults to 0.0) --
  The preemphasis coefficient.
- **preemphasis_htk_flavor** (`bool`, *optional*, defaults to `True`) --
  Whether to use HTK-style preemphasis.
- **fft_overdrive** (`bool`, *optional*, defaults to `False`) --
  Whether to use FFT overdrive.
- **dither** (`float`, *optional*, defaults to 0.0) --
  Adds dithering. In other words, adds a small Gaussian noise to each frame.
  E.g. use 0.0001 to add dithering with a normal distribution centered
  around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range of raw_speech).
  The value 0.0 means no dithering.
  Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
  the high log_mel_fbank values for signals with hard-zero sections,
  when VAD cutoff is present in the signal.
- **input_scale_factor** (`float`, *optional*, defaults to 1.0) --
  Scaling factor applied to the input waveform.
- **mel_floor** (`float`, *optional*, defaults to 0.001) --
  Minimum value for Mel spectrograms to avoid log(0).
- **per_bin_mean** (`Optional[Sequence[float]]`, *optional*) --
  Mean values for per-bin normalization.
- **per_bin_stddev** (`Optional[Sequence[float]]`, *optional*) --
  Standard deviation values for per-bin normalization.
An audio feature extractor Universal Speech Models https://huggingface.co/papers/2303.01037.

- **raw_speech** --
  The audio for which MEL spectrograms are created.
- **padding** (`Union[bool, str, PaddingStrategy]`, *optional*, defaults to `"longest"`) --
  The padding strategy to use for batches of audio with different lengths.
- **max_length** (`int`, *optional*, defaults to 480000) --
  If provided, defines the maximum length of the audio to allow. Audio longer than this will be
  truncated if `truncation=True`.
- **truncation** (`bool`, *optional*, defaults to `True`) --
  Whether or not to truncate audio above `max_length`.
- **pad_to_multiple_of** (`int`, *optional*, defaults to 128) --
  When padding, pad to a multiple of this value. The default value is defined for optimal TPU support.
- **return_tensors** (`Union[str, TensorType]`, *optional*, defaults to `None`) --
  The type of tensors to return (e.g., NumPy, or Torch).
- **return_attention_mask** (`bool`, *optional*, defaults to `True`) --
  Whether to return the attention mask for the generated MEL spectrograms.
Creates a batch of MEL spectrograms from the provided raw speech.

This implementation uses a different algorithm for windowing and preemphasis compared to the built-in
`transformers.audio_utils.spectrogram()` function that _will_ result in different outputs. Consider this
carefully when selecting an audio feature extractor, especially with pre-trained models.

## Gemma4ImageProcessorPil[[transformers.Gemma4ImageProcessorPil]]

- **patch_size** (`int`, *kwargs*, *optional*) --
  Size of each image patch in pixels.
- **max_soft_tokens** (`int`, *kwargs*, *optional*) --
  Maximum number of soft (vision) tokens per image.
  Must be one of {70, 140, 280, 560, 1120}.
- **pooling_kernel_size** (`int`, *kwargs*, *optional*) --
  Spatial pooling kernel size applied after patchification.
- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.
Constructs a Gemma4 image processor.

- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **patch_size** (`int`, *kwargs*, *optional*) --
  Size of each image patch in pixels.
- **max_soft_tokens** (`int`, *kwargs*, *optional*) --
  Maximum number of soft (vision) tokens per image.
  Must be one of {70, 140, 280, 560, 1120}.
- **pooling_kernel_size** (`int`, *kwargs*, *optional*) --
  Spatial pooling kernel size applied after patchification.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.`~image_processing_base.BatchFeature`- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## Gemma4ImageProcessor[[transformers.Gemma4ImageProcessor]]

- **patch_size** (`int`, *kwargs*, *optional*) --
  Size of each image patch in pixels.
- **max_soft_tokens** (`int`, *kwargs*, *optional*) --
  Maximum number of soft (vision) tokens per image.
  Must be one of {70, 140, 280, 560, 1120}.
- **pooling_kernel_size** (`int`, *kwargs*, *optional*) --
  Spatial pooling kernel size applied after patchification.
- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.
Constructs a Gemma4 image processor.

## Gemma4VideoProcessor[[transformers.Gemma4VideoProcessor]]

- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
- **size** (`dict`, *optional*, defaults to `self.size`) --
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
- **size_divisor** (`int`, *optional*, defaults to `self.size_divisor`) --
  The size by which to make sure both the height and width can be divided.
- **default_to_square** (`bool`, *optional*, defaults to `self.default_to_square`) --
  Whether to default to a square video when resizing, if size is an int.
- **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
- **crop_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) --
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
- **rescale_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) --
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
- **do_convert_rgb** (`bool`, *optional*, defaults to `self.image_std`) --
  Whether to convert the video to RGB.
- **video_metadata** (`VideoMetadata`, *optional*) --
  Metadata of the video containing information about total duration, fps and total number of frames.
- **do_sample_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) --
  Whether to sample frames from the video before processing or to process the whole video.
- **num_frames** (`int`, *optional*, defaults to `self.num_frames`) --
  Maximum number of frames to sample when `do_sample_frames=True`.
- **fps** (`int` or `float`, *optional*, defaults to `self.fps`) --
  Target frames to sample per second when `do_sample_frames=True`.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output video. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
  - Unset: Use the channel dimension format of the input video.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
- **device** (`torch.device`, *optional*) --
  The device to process the videos on. If unset, the device is inferred from the input videos.
- **return_metadata** (`bool`, *optional*) --
  Whether to return video metadata or not.
Constructs a Gemma4 video processor that samples frames from videos for use with the Gemma4 model.

## Gemma4Processor[[transformers.Gemma4Processor]]

- **feature_extractor** (*Gemma4AudioFeatureExtractor*) --
  The feature extractor is a required input.
- **image_processor** (*Gemma4ImageProcessor*) --
  The image processor is a required input.
- **tokenizer** (*tokenizer_class*) --
  The tokenizer is a required input.
- **video_processor** (*Gemma4VideoProcessor*) --
  The video processor is a required input.
- **chat_template** (*str*) --
  A Jinja template to convert lists of messages in a chat into a tokenizable string.
- **image_seq_length** (*int*, *optional*, defaults to 280) --
  The number of soft tokens per image used for placeholder expansion.
- **audio_seq_length** (*int*, *optional*, defaults to 750) --
  The maximum number of audio soft tokens per audio segment. Serves as an
  upper-bound cap when dynamic audio token counts are computed.
- **audio_ms_per_token** (*int*, *optional*, defaults to 40) --
  Milliseconds of audio per output soft token. Used to dynamically compute
  the number of audio placeholder tokens as `ceil(duration_ms / audio_ms_per_token)`.
  The default of 40 comes from the SSCP convolution's 4× time reduction on 10ms frames.
Constructs a Gemma4Processor which wraps a feature extractor, a image processor, a tokenizer, and a video processor into a single processor.

[*Gemma4Processor*] offers all the functionalities of [*Gemma4AudioFeatureExtractor*], [*Gemma4ImageProcessor*], [*tokenizer_class*], and [*Gemma4VideoProcessor*]. See the
[*~Gemma4AudioFeatureExtractor*], [*~Gemma4ImageProcessor*], [*~tokenizer_class*], and [*~Gemma4VideoProcessor*] for more information.

- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **text** (`Union[str, list[str], list[list[str]]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.
- **videos** (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) --
  Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
  passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
- **audio** (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) --
  The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor.
  In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
  and T is the sample length of the audio.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors of a particular framework. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return NumPy `np.ndarray` objects.
- ****kwargs** ([ProcessingKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) --
  Additional processing options for each modality (text, images, videos, audio). Model-specific parameters
  are listed above; see the TypedDict class for the complete list of supported arguments.

## Gemma4PreTrainedModel[[transformers.Gemma4PreTrainedModel]]

- **config** ([PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

Define the computation performed at every call.

Should be overridden by all subclasses.

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## Gemma4AudioModel[[transformers.Gemma4AudioModel]]

An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture.

- **input_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`) --
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Gemma4AudioFeatureExtractor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor). See [Gemma4AudioFeatureExtractor.__call__()](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor.__call__) for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4AudioFeatureExtractor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor) for processing audios).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)`tuple[torch.Tensor, torch.BoolTensor]`
Encodes audio features to soft tokens.

## Gemma4VisionModel[[transformers.Gemma4VisionModel]]

The Gemma 4 Vision Encoder.

- **pixel_values** (`torch.FloatTensor` or `list[torch.FloatTensor]`) --
  The images to encode. Either a single `[batch, channels, height, width]` tensor
  (all images same size) or a list of `[1, channels, height, width]` tensors (different sizes).
- **pixel_position_ids** (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`) --
  The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).[BaseModelOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`A [BaseModelOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) and inputs.
Encodes image pixels to soft tokens from patches.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Gemma4TextModel[[transformers.Gemma4TextModel]]

- **config** ([Gemma4TextConfig](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4TextConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
The base Gemma 4 language model without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **per_layer_inputs** (`torch.Tensor`, *optional*) --
  Pre-computed per-layer input text embeddings of shape `(batch_size, sequence_length, num_hidden_layers,
  hidden_size_per_layer_input)`. When provided, these are used directly instead of being computed from `input_ids`
  via `get_per_layer_inputs()` in the text model. If calling the `forward` with `inputs_embeds` instead of `input_ids`,
  you should probably precompute them and forward them along `inputs_embeds`, otherwise recomputing them needs
  to reverse the main embedding, which is expensive.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).`Gemma4TextModelOutputWithPast` or `tuple(torch.FloatTensor)`A `Gemma4TextModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) and inputs.
The [Gemma4TextModel](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **shared_kv_states** (`dict`, *optional*) -- Dictionary mapping layer type strings to tuples of (key_states, value_states) tensors.
  Used to pass shared KV states between layers during KV sharing.

## Gemma4ForCausalLM[[transformers.Gemma4ForCausalLM]]

- **config** ([Gemma4TextConfig](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4TextConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
The base Gemma 4 language model with a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **logits_to_keep** (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).
- **per_layer_inputs** (`torch.Tensor`, *optional*) --
  Pre-computed per-layer input text embeddings of shape `(batch_size, sequence_length, num_hidden_layers,
  hidden_size_per_layer_input)`. When provided, these are used directly instead of being computed from `input_ids`
  via `get_per_layer_inputs()` in the text model. If calling the `forward` with `inputs_embeds` instead of `input_ids`,
  you should probably precompute them and forward them along `inputs_embeds`, otherwise recomputing them needs
  to reverse the main embedding, which is expensive.`Gemma4CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`A `Gemma4CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) and inputs.
The [Gemma4ForCausalLM](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
- **shared_kv_states** (`dict`, *optional*) -- Dictionary mapping layer type strings to tuples of (key_states, value_states) tensors.
  Used to pass shared KV states between layers during KV sharing.

Example:

```python
>>> from transformers import AutoTokenizer, Gemma4ForCausalLM

>>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-2-9b")
>>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

>>> prompt = "What is your favorite condiment?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"What is your favorite condiment?"
```

## Gemma4Model[[transformers.Gemma4Model]]

- **config** ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The base Gemma 4 model comprising a vision backbone, an audio backbone, and a language model without a
language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Gemma4ImageProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4ImageProcessor). See `Gemma4ImageProcessor.__call__()` for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4ImageProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4ImageProcessor) for processing images).
- **pixel_values_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) --
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Gemma4VideoProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4VideoProcessor). See `Gemma4VideoProcessor.__call__()` for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4VideoProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4VideoProcessor) for processing videos).
- **input_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) --
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Gemma4AudioFeatureExtractor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor). See [Gemma4AudioFeatureExtractor.__call__()](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor.__call__) for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4AudioFeatureExtractor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor) for processing audios).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **input_features_mask** (`torch.FloatTensor]` of shape `(num_images, seq_length)`) --
  The attention mask for the input audio.
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **mm_token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2).
  Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **image_position_ids** (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*) --
  2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
  Passed through to the vision encoder for positional embedding computation.
- **video_position_ids** (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*) --
  2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
  Passed through to the vision encoder for positional embedding computation.
- **per_layer_inputs** (`torch.Tensor`, *optional*) --
  Pre-computed per-layer input text embeddings of shape `(batch_size, sequence_length, num_hidden_layers,
  hidden_size_per_layer_input)`. When provided, these are used directly instead of being computed from `input_ids`
  via `get_per_layer_inputs()` in the text model. If calling the `forward` with `inputs_embeds` instead of `input_ids`,
  you should probably precompute them and forward them along `inputs_embeds`, otherwise recomputing them needs
  to reverse the main embedding, which is expensive.`Gemma4ModelOutputWithPast` or `tuple(torch.FloatTensor)`A `Gemma4ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) and inputs.
The [Gemma4Model](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
- **shared_kv_states** (`dict`, *optional*) -- Dictionary mapping layer type strings to tuples of (key_states, value_states) tensors.
  Used to pass shared KV states between layers during KV sharing.

## Gemma4ForConditionalGeneration[[transformers.Gemma4ForConditionalGeneration]]

- **config** ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The base Gemma 4 model comprising a vision backbone, an audio backbone, a language model, and a language modeling
head.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Gemma4ImageProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4ImageProcessor). See `Gemma4ImageProcessor.__call__()` for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4ImageProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4ImageProcessor) for processing images).
- **pixel_values_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) --
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Gemma4VideoProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4VideoProcessor). See `Gemma4VideoProcessor.__call__()` for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4VideoProcessor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4VideoProcessor) for processing videos).
- **input_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) --
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Gemma4AudioFeatureExtractor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor). See [Gemma4AudioFeatureExtractor.__call__()](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor.__call__) for details ([Gemma4Processor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Processor) uses
  [Gemma4AudioFeatureExtractor](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor) for processing audios).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **input_features_mask** (`torch.FloatTensor]` of shape `(num_images, seq_length)`) --
  The attention mask for the input audio.
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **image_position_ids** (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*) --
  2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
  Passed through to the vision encoder for positional embedding computation.
- **video_position_ids** (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*) --
  2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
  Passed through to the vision encoder for positional embedding computation.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **mm_token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2).
  Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **logits_to_keep** (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).
- **per_layer_inputs** (`torch.Tensor`, *optional*) --
  Pre-computed per-layer input text embeddings of shape `(batch_size, sequence_length, num_hidden_layers,
  hidden_size_per_layer_input)`. When provided, these are used directly instead of being computed from `input_ids`
  via `get_per_layer_inputs()` in the text model. If calling the `forward` with `inputs_embeds` instead of `input_ids`,
  you should probably precompute them and forward them along `inputs_embeds`, otherwise recomputing them needs
  to reverse the main embedding, which is expensive.`Gemma4CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`A `Gemma4CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma4Config](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4Config)) and inputs.
The [Gemma4ForConditionalGeneration](/docs/transformers/main/en/model_doc/gemma4#transformers.Gemma4ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
- **shared_kv_states** (`dict`, *optional*) -- Dictionary mapping layer type strings to tuples of (key_states, value_states) tensors.
  Used to pass shared KV states between layers during KV sharing.

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, Gemma4ForConditionalGeneration

>>> model = Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-e2b-it")
>>> processor = AutoProcessor.from_pretrained("google/gemma-4-e2b-it")

>>> messages = [
...     {
...         "role": "user", "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "Where is the cat standing?"},
...         ]
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     add_generation_prompt=True
... )
>>> # Generate
>>> generate_ids = model.generate(**inputs)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
```

