/**
 * Pull a readable message out of an exo API error body.
 *
 * exo reports errors OpenAI-style as `{"error": {"message": ...}}`, both as
 * HTTP error bodies and as `data:` chunks mid-stream. FastAPI's own handlers
 * use `{"detail": "..."}`. Returns null when the body has neither.
 */
export function extractApiErrorMessage(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const { error, detail } = body as { error?: unknown; detail?: unknown };
  if (error && typeof error === "object") {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message) return message;
  }
  if (typeof detail === "string" && detail) return detail;
  return null;
}

/**
 * Read a failed response's body and return a readable error message,
 * falling back to the raw body text (or the status line if it is empty).
 */
export async function readApiErrorMessage(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const message = extractApiErrorMessage(JSON.parse(text));
    if (message) return message;
  } catch {
    // Not JSON; fall back to the raw body
  }
  return text || `${response.status} ${response.statusText}`;
}
