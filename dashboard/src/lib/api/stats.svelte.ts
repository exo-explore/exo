import { browser } from "$app/environment";

export interface ServerStats {
  uptimeSeconds: number;
  totalRequests: number;
  instanceCount: number;
  nodeCount: number;
  activeCommands: number;
}

const POLL_MS = 2000;

function createServerStatsStore() {
  let value = $state<ServerStats | null>(null);
  let lastError = $state<string | null>(null);
  let started = false;

  async function fetchOnce() {
    try {
      const res = await fetch("/v1/stats");
      if (!res.ok) throw new Error(`status ${res.status}`);
      value = (await res.json()) as ServerStats;
      lastError = null;
    } catch (err) {
      value = null;
      lastError = err instanceof Error ? err.message : String(err);
    }
  }

  function start() {
    if (started || !browser) return;
    started = true;
    fetchOnce();
    setInterval(fetchOnce, POLL_MS);
  }

  if (browser) start();

  return {
    get value() {
      return value;
    },
    get lastError() {
      return lastError;
    },
  };
}

export const serverStats = createServerStatsStore();
