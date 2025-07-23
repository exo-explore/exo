from . import (
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
)

TaskSagaEvent = (
        MLXInferenceSagaPrepare
        | MLXInferenceSagaStartPrepare
)
