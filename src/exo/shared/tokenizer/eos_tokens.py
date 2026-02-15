from exo.shared.models.model_cards import ModelId


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    model_id_lower = model_id.lower()

    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-4.7-flash" in model_id_lower:
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        return [151336, 151329, 151338]

    return None
