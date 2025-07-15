from typing import Literal, TypedDict

from master.sanity_checking import check_keys_in_map_match_enum_values
from shared.types.events.common import EventCategoryEnum, State


class SyncStateManagerMapping(TypedDict):
    MutatesTaskState: State[Literal[EventCategoryEnum.MutatesTaskState]]
    MutatesTaskSagaState: State[Literal[EventCategoryEnum.MutatesTaskSagaState]]
    MutatesControlPlaneState: State[Literal[EventCategoryEnum.MutatesControlPlaneState]]
    MutatesDataPlaneState: State[Literal[EventCategoryEnum.MutatesDataPlaneState]]
    MutatesRunnerStatus: State[Literal[EventCategoryEnum.MutatesRunnerStatus]]
    MutatesInstanceState: State[Literal[EventCategoryEnum.MutatesInstanceState]]
    MutatesNodePerformanceState: State[
        Literal[EventCategoryEnum.MutatesNodePerformanceState]
    ]


check_keys_in_map_match_enum_values(SyncStateManagerMapping, EventCategoryEnum)
