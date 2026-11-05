"""Request content (prompts, replies, tool arguments) stays out of EXO's logs.

A log line about a request names it by id and says how big it was: lengths, token counts and
ids, never text. People send EXO private material, and a log file outlives the request and is
copied by bug reports, so the text of a request isn't something to write to one by default.

Set `EXO_LOG_REQUEST_CONTENT=true` to bring the text back for debugging. It is read on every
call, so it can be switched with the environment of a running test. Leave it off anywhere
the requests aren't yours.
"""

import os
from typing import Final

LOG_REQUEST_CONTENT_ENV: Final = "EXO_LOG_REQUEST_CONTENT"


def logging_request_content() -> bool:
    """Whether request content may be written to logs (off unless asked for)."""
    return os.environ.get(LOG_REQUEST_CONTENT_ENV, "false").lower() == "true"


def describe_text(text: str) -> str:
    """Text as a log line may show it: its length, or the text itself with the debug flag."""
    return repr(text) if logging_request_content() else f"<{len(text)} chars>"
