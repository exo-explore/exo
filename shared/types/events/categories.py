
from shared.types.events.events import (
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
)

TaskSagaEvent = (
    MLXInferenceSagaPrepare
    | MLXInferenceSagaStartPrepare
)