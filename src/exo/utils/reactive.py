"""
Utilities for reactive variables

"""

from typing import Protocol


class OnChange[T](Protocol):
    def __call__(self, old_value: T, new_value: T) -> None: ...


class Reactive[T]:
    def __init__(self, initial_value: T, on_change: OnChange[T]):
        self._value = initial_value
        self._on_change = on_change

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: T):
        old_value = self._value
        self._value = new_value

        # don't notify when not changed
        if old_value == new_value:
            return

        # notify of changes
        self._on_change(old_value=old_value, new_value=new_value)
