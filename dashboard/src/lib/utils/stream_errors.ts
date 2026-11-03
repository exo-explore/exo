/**
 * Turn the error that ended a streamed reply into one short sentence for the
 * chat. The server may send a runner's whole traceback ("Runner shutdown before
 * completing command (exitcode=1\nRunner error: …\nTraceback …"); people only
 * need to know what happened, so it is never shown as it is.
 */

/** What was being generated, for the wording ("before the reply finished"). */
export type StreamKind = "reply" | "image";

// A runner fails this way when another runner of its instance stops: the ring
// or JACCL connection to it breaks, usually because its device left.
const LOST_PEER =
  /connection to a peer was lost|lost the connection to the instance's other runners|ring socket receive failed|ring transport aborted|ECONNRESET|\[(ring|jaccl)\]/i;

// What browsers throw when the connection to the server drops mid-request
const NETWORK_FAILURE =
  /^(failed to fetch|load failed|network ?error|networkerror when attempting to fetch resource\.?|the network connection was lost\.?)$/i;

const RUNNER_SHUTDOWN_PREFIX = "Runner shutdown before completing command";
const RUNNER_ERROR_PREFIX = "Runner error:";
// A Python exception's class, e.g. "RuntimeError: "
const EXCEPTION_CLASS = /^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit): /;
const MAX_LENGTH = 200;

function sentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return trimmed;
  const capped =
    trimmed.length > MAX_LENGTH
      ? `${trimmed.slice(0, MAX_LENGTH - 1).trimEnd()}…`
      : trimmed;
  return /[.!?…]$/.test(capped) ? capped : `${capped}.`;
}

/**
 * One short, plain sentence for an error that ended a stream.
 *
 * @param message the error's message (from the server's error chunk, the
 *   HTTP error body, or the browser's network error)
 * @param kind what was being generated
 */
export function describeStreamError(
  message: string,
  kind: StreamKind = "reply",
): string {
  const lines = message
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const first = lines[0] ?? "";
  const unfinished = `before the ${kind} finished`;

  if (lines.length === 0) return `Something went wrong ${unfinished}.`;
  if (NETWORK_FAILURE.test(first)) {
    return `Lost the connection to exo ${unfinished}.`;
  }
  if (lines.some((line) => LOST_PEER.test(line))) {
    return `A device left the cluster ${unfinished}.`;
  }

  if (first.startsWith(RUNNER_SHUTDOWN_PREFIX)) {
    const runnerError = lines.find((line) =>
      line.startsWith(RUNNER_ERROR_PREFIX),
    );
    if (!runnerError) {
      return `The model stopped unexpectedly ${unfinished}.`;
    }
    let reason = runnerError.slice(RUNNER_ERROR_PREFIX.length).trim();
    reason = reason.replace(EXCEPTION_CLASS, "");
    // "…\nRunner error: <reason>)" closes the "(" opened on the first line
    if (lines[lines.length - 1] === runnerError && reason.endsWith(")")) {
      reason = reason.slice(0, -1).trimEnd();
    }
    return sentence(`The model stopped ${unfinished}: ${reason}`);
  }

  // Anything else: its first line, without a traceback or a Python class name
  return sentence(first.replace(EXCEPTION_CLASS, ""));
}
