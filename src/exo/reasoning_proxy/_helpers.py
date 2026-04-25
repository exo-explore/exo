from typing import cast


def as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def as_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return cast(list[object], value)
    return None


def as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def dict_get_str(d: dict[str, object], key: str) -> str | None:
    return as_str(d.get(key))


def dict_get_list(d: dict[str, object], key: str) -> list[object] | None:
    return as_list(d.get(key))


def dict_get_dict(d: dict[str, object], key: str) -> dict[str, object] | None:
    return as_dict(d.get(key))
