class Request:
    request_id: str
    prompt_token_ids: list[int] | None
    num_prompt_tokens: int
    num_computed_tokens: int
    num_tokens: int
