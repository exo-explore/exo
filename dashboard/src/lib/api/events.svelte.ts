import { browser } from "$app/environment";

export interface TickerEvent {
  at: number;
  kind: string;
  label?: string;
  detail?: string;
}

interface TraceItem {
  task_id: string;
  created_at: string;
  file_size: number;
}

const MAX_EVENTS = 12;
const POLL_MS = 3000;

function createRecentEventsStore() {
  let value = $state<TickerEvent[]>([]);
  let started = false;
  const seen = new Set<string>();

  function push(ev: TickerEvent) {
    value = [ev, ...value].slice(0, MAX_EVENTS);
  }

  async function poll() {
    try {
      const res = await fetch("/v1/traces");
      if (!res.ok) return;
      const data = (await res.json()) as { traces: TraceItem[] };
      const traces = data.traces ?? [];

      // Seed `seen` on first run so we don't flood the ticker with history.
      if (seen.size === 0 && traces.length > 0) {
        for (const t of traces) seen.add(t.task_id);
        // Surface the most recent few as "history" so the ticker isn't empty.
        const recent = traces.slice(0, 4).reverse();
        for (const t of recent) {
          push({
            at: new Date(t.created_at).getTime(),
            kind: "trace",
            label: t.task_id.slice(0, 10),
            detail: `${(t.file_size / 1024).toFixed(1)} KB`,
          });
        }
        return;
      }

      // Surface new traces in chronological order.
      const fresh = traces.filter((t) => !seen.has(t.task_id)).reverse();
      for (const t of fresh) {
        seen.add(t.task_id);
        push({
          at: new Date(t.created_at).getTime(),
          kind: "trace",
          label: t.task_id.slice(0, 10),
          detail: `${(t.file_size / 1024).toFixed(1)} KB`,
        });
      }
    } catch {
      // ignore
    }
  }

  function start() {
    if (started || !browser) return;
    started = true;
    poll();
    setInterval(poll, POLL_MS);
  }

  if (browser) start();

  return {
    get value() {
      return value;
    },
  };
}

export const recentEvents = createRecentEventsStore();
