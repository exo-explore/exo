import asyncio
from typing import Any, Callable
from exo.helpers import AsyncCallbackSystem, AsyncCallback


# Usage example
async def main() -> None:
  callback_system = AsyncCallbackSystem[str, Any]()

  # Register callbacks
  callback1 = callback_system.register("callback1")
  callback2 = callback_system.register("callback2")

  def on_next_callback(name: str) -> Callable[..., None]:
    def callback(*args: Any) -> None:
      print(f"{name} received values: {args}")

    return callback

  callback1.on_next(on_next_callback("Callback1"))
  callback2.on_next(on_next_callback("Callback2"))

  async def wait_for_callback(name: str, callback: AsyncCallback[Any], condition: Callable[..., bool]) -> None:
    try:
      result = await callback.wait(condition, timeout=2)
      print(f"{name} wait completed with result: {result}")
    except asyncio.TimeoutError:
      print(f"{name} wait timed out")

  # Trigger all callbacks at once
  callback_system.trigger_all("Hello", 42, True)

  # Wait for all callbacks with different conditions
  await asyncio.gather(
    wait_for_callback("Callback1", callback1, lambda msg, num, flag: isinstance(msg, str) and num > 0),
    wait_for_callback("Callback2", callback2, lambda msg, num, flag: flag is True),
  )

  # Trigger individual callback
  callback_system.trigger("callback2", "World", -10, False)

  # Demonstrate timeout
  new_callback = callback_system.register("new_callback")
  new_callback.on_next(on_next_callback("NewCallback"))
  await wait_for_callback("NewCallback", new_callback, lambda msg, num, flag: num > 100)

  callback_system.trigger("callback2", "World", 200, False)


asyncio.run(main())
