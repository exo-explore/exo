from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Iterable, Optional, Protocol, Sequence, cast, final, runtime_checkable

import chainlit as cl
import chainlit.callbacks as callbacks
from chainlit.action import Action
from chainlit.context import context
from chainlit.message import ErrorMessage
from chainlit.chat_context import chat_context
from exo.shared.models.model_cards import MODEL_CARDS
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")

MODELS = list(MODEL_CARDS.keys())


@final
class ThreadModelStore:
    """Per-thread selected model registry.

    Chainlit provides a distinct thread ID per conversation; we map that ID to
    the chosen model for subsequent completions.
    """

    _by_thread_id: Dict[str, str] = {}

    @classmethod
    def set_model(cls, thread_id: str, model_name: str) -> None:
        cls._by_thread_id[thread_id] = model_name

    @classmethod
    def get_model(cls, thread_id: str, default_model: str) -> str:
        return cls._by_thread_id.get(thread_id, default_model)

# Provide typed decorator aliases to satisfy strict type checking.
if TYPE_CHECKING:
    def on_chat_start_dec(func: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]: ...
    def on_message_dec(func: Callable[[cl.Message], Awaitable[None]]) -> Callable[[cl.Message], Awaitable[None]]: ...
    def action_callback_dec(name: str) -> Callable[[Callable[[Action], Awaitable[None]]], Callable[[Action], Awaitable[None]]]: ...
else:
    on_chat_start_dec = callbacks.on_chat_start
    on_message_dec = callbacks.on_message
    action_callback_dec = callbacks.action_callback

@on_chat_start_dec
async def start() -> None:
    # Render model choices inline on the main screen as action buttons.
    actions = [Action(name="set_model", payload={}, label=m) for m in MODELS]
    await cl.Message(
        content="Choose a model to use for this conversation:",
        actions=actions,
    ).send()

@action_callback_dec("set_model")
async def on_model_selected(action: Action) -> None:
    thread_id = context.session.thread_id
    selected_model = action.label if action.label in MODELS else MODELS[0]

    ThreadModelStore.set_model(thread_id, selected_model)
    await cl.Message(f"Model set to: {selected_model}").send()

@on_message_dec
async def on_message(msg: cl.Message) -> None:
    """Handle user messages by calling the chat completions API.

    Any API error is transformed into an inline error message for the user.
    """
    thread_id = context.session.thread_id
    model = ThreadModelStore.get_model(thread_id, MODELS[0])
    user_text = cast("str | None", getattr(msg, "content", None)) or ""
    history_raw: list[dict[str, str]] = cast(list[dict[str, str]], chat_context.to_openai())
    history_raw.append({"role": "user", "content": user_text})

    def to_openai_params(items: list[dict[str, str]]) -> list[ChatCompletionMessageParam]:
        return cast(list[ChatCompletionMessageParam], items)

    history = to_openai_params(history_raw)
    assistant_msg: cl.Message = cl.Message(content="")
    
    @runtime_checkable
    class _HasContent(Protocol):
        content: Optional[str]

    @runtime_checkable
    class _ChoiceDelta(Protocol):
        delta: Optional[_HasContent]
        message: Optional[_HasContent]

    @runtime_checkable
    class _Chunk(Protocol):
        choices: Sequence[_ChoiceDelta]
    try:
        stream_any: object = client.chat.completions.create(
            model=model,
            messages=history,
            stream=True,
        )
        stream = cast(Iterable[_Chunk], stream_any)
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            token: Optional[str] = None
            if getattr(choice, "delta", None) and getattr(choice.delta, "content", None):
                token = choice.delta.content  # type: ignore[attr-defined]
            if token:
                # stream_token exists on Message; ignore type checker limitations
                await assistant_msg.stream_token(token)
        await assistant_msg.send()
    except Exception as e:
        if "404" in str(e):
            await ErrorMessage(
                (
                    f"No instance found for model {model}. You need to load an instance of {model} "
                    "first on the EXO dashboard: http://localhost:8000"
                )
            ).send()
            return
        await ErrorMessage(f"Request failed: {e!s}").send()
