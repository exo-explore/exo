import torch

def adjust_audio_features_to_expected_length(
    audio_features: torch.Tensor, expected_tokens: int, audio_padding_embs: torch.Tensor
) -> tuple[torch.Tensor, int]: ...
