from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


class ReasoningParserState(Enum):
    """State constants for reasoning parser streaming operations."""

    NORMAL = "normal"
    FOUND_PREFIX = "found_prefix"


class AbstractReasoningParser:
    """Abstract reasoning parser class that should not be used directly.

    Provided properties and methods should be used in derived classes to parse
    reasoning content from model outputs.
    """

    def __init__(
        self,
        reasoning_open: str,
        reasoning_close: str,
        state: ReasoningParserState = ReasoningParserState.NORMAL,
    ) -> None:
        """Initialize the reasoning parser.

        Parameters
        ----------
        reasoning_open : str
            Opening tag/marker for reasoning content.
        reasoning_close : str
            Closing tag/marker for reasoning content.
        state : ReasoningParserState, optional
            Initial parser state, by default ReasoningParserState.NORMAL.
        """
        self.reasoning_open = reasoning_open
        self.reasoning_close = reasoning_close
        self.state = state
        self.buffer = ""

    def get_reasoning_open(self) -> str:
        """Get the opening tag for reasoning content.

        Returns
        -------
        str
            The opening tag string.
        """
        return self.reasoning_open

    def get_reasoning_close(self) -> str:
        """Get the closing tag for reasoning content.

        Returns
        -------
        str
            The closing tag string.
        """
        return self.reasoning_close

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.

        Returns
        -------
        bool
            True if the reasoning parser needs a redacted reasoning prefix, False otherwise.
        """
        return False

    def has_special_parsing(self) -> bool:
        """Check if the reasoning parser has special parsing logic.

        Returns
        -------
        bool
            True if the reasoning parser has special parsing logic, False otherwise.
        """
        return False

    def respects_enable_thinking(self) -> bool:
        """Check if the reasoning parser respects the enable_thinking flag.

        Returns
        -------
        bool
            True if the reasoning parser respects the enable_thinking flag, False otherwise.
        """
        return False

    def extract_reasoning(self, model_output: str) -> Dict[str, str] | None:
        """Extract reasoning content from complete model output.

        Parameters
        ----------
        model_output : str
            Complete model output to parse.

        Returns
        -------
        dict[str, str] | None
            Dictionary with 'reasoning' key containing extracted content,
            or None if no reasoning found.

        Raises
        ------
        NotImplementedError
            This method must be implemented by derived classes.
        """
        raise NotImplementedError(
            "AbstractReasoningParser.extract_reasoning has not been implemented!"
        )

    def extract_reasoning_streaming(
        self, chunk: str
    ) -> Tuple[Dict[str, str] | str | None, bool]:
        """Extract reasoning content from streaming chunks.

        Parameters
        ----------
        chunk : str
            Chunk of model output to process.

        Returns
        -------
        tuple[dict[str, str] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Reasoning dict, passthrough chunk, or None
            - is_complete: True if chunk should be sent, False if buffering

        Raises
        ------
        NotImplementedError
            This method must be implemented by derived classes.
        """
        raise NotImplementedError(
            "AbstractReasoningParser.extract_reasoning_streaming has not been implemented!"
        )


class ToolParserState(Enum):
    """State constants for tool parser streaming operations."""

    NORMAL = "normal"
    FOUND_PREFIX = "found_prefix"
    FOUND_ARGUMENTS = "found_arguments"


class AbstractToolParser:
    """Abstract tool parser class that should not be used directly.

    Provided properties and methods should be used in derived classes to parse
    tool calls from model outputs.
    """

    def __init__(
        self,
        tool_open: str,
        tool_close: str,
        state: ToolParserState = ToolParserState.NORMAL,
    ) -> None:
        """Initialize the tool parser.

        Parameters
        ----------
        tool_open : str
            Opening tag/marker for tool calls.
        tool_close : str
            Closing tag/marker for tool calls.
        state : ToolParserState, optional
            Initial parser state, by default ToolParserState.NORMAL.
        """
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.state = state
        self.buffer = ""

    def get_tool_open(self) -> str:
        """Get the opening tag for tool calls.

        Returns
        -------
        str
            The opening tag string.
        """
        return self.tool_open

    def get_tool_close(self) -> str:
        """Get the closing tag for tool calls.

        Returns
        -------
        str
            The closing tag string.
        """
        return self.tool_close

    def extract_tool_calls(self, model_output: str) -> Dict[str, List] | None:
        """Extract tool calls from complete model output.

        Parameters
        ----------
        model_output : str
            Complete model output to parse.

        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of tool calls,
            or None if no tool calls found.

        Raises
        ------
        NotImplementedError
            This method must be implemented by derived classes.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!"
        )

    def extract_tool_calls_streaming(
        self, chunk: str
    ) -> Tuple[Dict[str, List] | str | None, bool]:
        """Extract tool calls from streaming chunks.

        Default implementation that buffers content between tool_open and tool_close
        tags. Subclasses can override this for custom streaming behavior.

        Parameters
        ----------
        chunk : str
            Chunk of model output to process.

        Returns
        -------
        tuple[dict[str, list] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Tool calls dict, passthrough chunk, or None
            - is_complete: True if chunk should be sent, False if buffering
        """
        self.buffer += chunk

        if self.state == ToolParserState.NORMAL:
            if self.tool_open in self.buffer:
                open_idx = self.buffer.find(self.tool_open)

                if open_idx > 0:
                    # Emit content before the tool call
                    to_emit = self.buffer[:open_idx]
                    self.buffer = self.buffer[open_idx:]
                    return {"content": to_emit}, True

                # Found prefix at start of buffer
                self.state = ToolParserState.FOUND_PREFIX
                # Fall through to check for close tag in the same buffer
            else:
                # Check for partial prefix at end of buffer
                partial_match_len = 0
                check_len = min(len(self.buffer), len(self.tool_open) - 1)

                for i in range(check_len, 0, -1):
                    if self.tool_open.startswith(self.buffer[-i:]):
                        partial_match_len = i
                        break

                if partial_match_len > 0:
                    # Buffer ends with partial prefix. Emit the rest.
                    to_emit = self.buffer[:-partial_match_len]
                    self.buffer = self.buffer[-partial_match_len:]
                    if to_emit:
                        return {"content": to_emit}, True
                    return None, False
                else:
                    # No partial prefix. Emit everything.
                    to_emit = self.buffer
                    self.buffer = ""
                    return {"content": to_emit}, True

        if self.state == ToolParserState.FOUND_PREFIX:
            if self.tool_close in self.buffer:
                close_idx = self.buffer.find(self.tool_close) + len(self.tool_close)

                full_tool_call = self.buffer[:close_idx]
                remainder = self.buffer[close_idx:]

                result = self.extract_tool_calls(full_tool_call)

                self.buffer = remainder
                self.state = ToolParserState.NORMAL
                return result, True

            return None, False

        # Fallback (should be covered by NORMAL state logic)
        return {"content": chunk}, True
