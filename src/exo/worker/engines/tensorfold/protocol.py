from dataclasses import dataclass
from itertools import pairwise

from exo.api.types import GenerationStats, LogprobsContentItem, ToolCallItem, Usage
from exo.shared.types.chunks import Chunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.worker.engines.tensorfold.sse import ProtocolError
from exo.worker.engines.tensorfold.wire import (
    Choice,
    Envelope,
    ErrorFrame,
    RuntimeStats,
    SemanticFrame,
    StreamAnchor,
    TokenFrame,
    ToolDelta,
)


@dataclass
class _Tool:
    identifier: str = ""
    name: str = ""
    arguments: str = ""


class StreamProtocol:
    def __init__(self, model_id: ModelId, token_limit: int = 131072):
        self.model_id = model_id
        self._token_limit = token_limit
        self._response_id: str | None = None
        self._tokens: list[int] = []
        self._tools: dict[int, _Tool] = {}
        self._tool_bytes = 0
        self._finish: str | None = None
        self._runtime: RuntimeStats | None = None
        self._usage: Usage | None = None
        self._last_logprob_position = 0
        self.done = False

    def _identity(self, response_id: str, model: str) -> None:
        if model != str(self.model_id):
            raise ProtocolError("TensorFold changed model identity within the stream")
        if self._response_id is None:
            self._response_id = response_id
        elif response_id != self._response_id:
            raise ProtocolError(
                "TensorFold changed response identity within the stream"
            )

    def feed(self, data: str) -> list[Chunk]:
        if self.done:
            raise ProtocolError("TensorFold sent data after the end of its stream")
        if data == "[DONE]":
            self._validate_finish()
            self.done = True
            return []
        envelope = Envelope.model_validate_json(data)
        if envelope.error is not None:
            raise ProtocolError(ErrorFrame.model_validate_json(data).error.message)
        if envelope.object == "tensorfold.token.chunk":
            frame = TokenFrame.model_validate_json(data)
            self._identity(frame.id, frame.model)
            if self._finish is not None or frame.offset != len(self._tokens):
                raise ProtocolError(
                    "TensorFold token stream has a discontinuous offset"
                )
            if any(token < 0 or token > 0xFFFFFFFF for token in frame.token_ids):
                raise ProtocolError("TensorFold returned an invalid token identifier")
            if len(self._tokens) + len(frame.token_ids) > self._token_limit:
                raise ProtocolError("TensorFold reply exceeds the token stream limit")
            self._tokens.extend(frame.token_ids)
            return []
        frame = SemanticFrame.model_validate_json(data)
        self._identity(frame.id, frame.model)
        self._validate_anchor(frame.tensorfold_stream)
        if len(frame.choices) > 1:
            raise ProtocolError("TensorFold returned more than one completion choice")
        if frame.usage is not None:
            self._usage = frame.usage.as_exo()
        if frame.tensorfold is not None:
            self._runtime = frame.tensorfold
        chunks: list[Chunk] = []
        for choice in frame.choices:
            if self._finish is not None:
                raise ProtocolError(
                    "TensorFold sent completion data after its finish reason"
                )
            chunks.extend(self._choice(choice, frame.tensorfold_stream))
            if choice.finish_reason is not None:
                self._finish = choice.finish_reason
        return chunks

    def _validate_anchor(self, anchor: StreamAnchor) -> None:
        last = self._tokens[-1] if self._tokens else None
        if anchor.token_count != len(self._tokens) or anchor.token_id != last:
            raise ProtocolError(
                "TensorFold semantic anchor disagrees with committed tokens"
            )

    def _choice(self, choice: Choice, anchor: StreamAnchor) -> list[Chunk]:
        delta = choice.delta
        if delta.tool_calls is not None:
            if delta.tool_calls and anchor.token_id is None:
                raise ProtocolError("TensorFold emitted a tool call before a token")
            for tool in delta.tool_calls:
                self._append_tool(tool)
        output: list[Chunk] = []
        if delta.reasoning_content:
            output.append(self._text(delta.reasoning_content, anchor, thinking=True))
        if choice.logprobs is not None and choice.logprobs.content:
            rows = choice.logprobs.content
            positions = (
                anchor.logprob_positions.content
                if anchor.logprob_positions is not None
                else []
            )
            if len(positions) != len(rows):
                raise ProtocolError(
                    "TensorFold log probabilities lack exact token positions"
                )
            # A token the answer shows in parts gives one entry a part, all at its position
            ordered = pairwise([self._last_logprob_position, *positions])
            if any(later < earlier for earlier, later in ordered) or (
                positions[-1] >= len(self._tokens)
            ):
                raise ProtocolError(
                    "TensorFold log probabilities disagree with token order"
                )
            self._last_logprob_position = positions[-1]
            chunk = self._text(delta.content or "", anchor, thinking=False)
            output.append(
                chunk.model_copy(
                    update={
                        "logprobs_content": [
                            LogprobsContentItem(
                                token=row.token,
                                logprob=row.logprob,
                                bytes=row.bytes,
                                top_logprobs=row.top_logprobs,
                            )
                            for row in rows
                        ]
                    }
                )
            )
        elif delta.content:
            output.append(self._text(delta.content, anchor, thinking=False))
        return output

    def _text(self, text: str, anchor: StreamAnchor, *, thinking: bool) -> TokenChunk:
        if anchor.token_id is None:
            raise ProtocolError("TensorFold emitted text before committing a token")
        return TokenChunk(
            model=self.model_id,
            text=text,
            token_id=anchor.token_id,
            usage=None,
            is_thinking=thinking,
        )

    def _append_tool(self, delta: ToolDelta) -> None:
        tool = self._tools.setdefault(delta.index, _Tool())
        if delta.id is not None:
            if tool.identifier and tool.identifier != delta.id:
                raise ProtocolError("TensorFold changed a tool call's identifier")
            if not tool.identifier:
                self._tool_bytes += len(delta.id.encode())
            tool.identifier = delta.id
        name = delta.function.name or ""
        arguments = delta.function.arguments or ""
        self._tool_bytes += len(name.encode()) + len(arguments.encode())
        if self._tool_bytes > 1024 * 1024:
            raise ProtocolError("TensorFold tool calls exceed 1 MiB")
        tool.name += name
        tool.arguments += arguments

    def _validate_finish(self) -> None:
        if self._finish is None or self._runtime is None or self._usage is None:
            raise ProtocolError("TensorFold stream ended without final metadata")
        if self._runtime.token_ids != self._tokens:
            raise ProtocolError(
                "TensorFold final token identifiers differ from its stream"
            )
        usage = self._usage
        if usage.completion_tokens != len(self._tokens):
            raise ProtocolError(
                "TensorFold completion count differs from its committed tokens"
            )
        if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
            raise ProtocolError("TensorFold returned inconsistent token usage")
        if not 0 <= usage.prompt_tokens_details.cached_tokens <= usage.prompt_tokens:
            raise ProtocolError("TensorFold returned inconsistent cached-token usage")
        if not (
            0
            <= usage.completion_tokens_details.reasoning_tokens
            <= usage.completion_tokens
        ):
            raise ProtocolError(
                "TensorFold returned inconsistent reasoning-token usage"
            )
        if self._finish == "tool_calls":
            if not self._tools or sorted(self._tools) != list(range(len(self._tools))):
                raise ProtocolError("TensorFold returned incomplete tool call indexes")
            if any(
                not tool.identifier or not tool.name for tool in self._tools.values()
            ):
                raise ProtocolError("TensorFold returned an incomplete tool call")
        elif self._tools:
            raise ProtocolError(
                "TensorFold emitted tool calls without a tool-call finish"
            )

    def finish(self, peak_memory_bytes: int) -> Chunk:
        if not self.done:
            raise ProtocolError("TensorFold stream ended without its completion marker")
        self._validate_finish()
        assert self._usage is not None and self._runtime is not None
        usage, runtime = self._usage, self._runtime
        cached = usage.prompt_tokens_details.cached_tokens
        prefill = runtime.prefill_seconds
        stats = GenerationStats(
            prompt_tps=(usage.prompt_tokens - cached) / prefill if prefill else 0.0,
            generation_tps=runtime.tokens_per_second,
            prompt_tokens=usage.prompt_tokens,
            generation_tokens=usage.completion_tokens,
            peak_memory_usage=Memory.from_bytes(peak_memory_bytes),
            prefix_cache_hit="exact"
            if cached == usage.prompt_tokens and cached
            else "partial"
            if cached
            else "none",
        )
        if self._finish == "tool_calls":
            return ToolCallChunk(
                model=self.model_id,
                tool_calls=[
                    ToolCallItem(
                        id=tool.identifier, name=tool.name, arguments=tool.arguments
                    )
                    for _, tool in sorted(self._tools.items())
                ],
                usage=usage,
                stats=stats,
            )
        assert self._finish in ("stop", "length", "content_filter")
        return TokenChunk(
            model=self.model_id,
            text="",
            token_id=self._tokens[-1] if self._tokens else None,
            usage=usage,
            stats=stats,
            finish_reason=self._finish,
        )
