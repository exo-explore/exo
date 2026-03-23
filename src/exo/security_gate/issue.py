from typing import Literal, NamedTuple, final


@final
class Issue(NamedTuple):
    filepath: str
    lineno: int
    check_id: str
    message: str
    severity: Literal["block", "advisory"]
