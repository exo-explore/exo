from _typeshed import Incomplete

class EngineGenerateError(Exception): ...

class EngineDeadError(Exception):
    __suppress_context__: Incomplete
    def __init__(self, *args, suppress_context: bool = False, **kwargs) -> None: ...
