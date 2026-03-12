import torch

def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int: ...
def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor: ...
def compute_mrope_for_media(
    video_size_thw: torch.LongTensor,
    spatial_merge_size: int,
    tokens_per_second: float = 1.0,
    video_second_per_grid: float = 1.0,
) -> torch.Tensor: ...
def recompute_mrope_positions(
    input_ids: torch.LongTensor,
    multimodal_positions: list[torch.Tensor],
    mrope_positions: torch.LongTensor,
    num_computed_tokens: int,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int,
) -> tuple[torch.LongTensor, int]: ...
