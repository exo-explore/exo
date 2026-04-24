<script lang="ts">
  import { recentEvents } from "$lib/api/events.svelte";

  let events = $derived(recentEvents.value);

  function timeOf(d: number): string {
    const t = new Date(d);
    return t.toTimeString().slice(0, 8);
  }

  // Repeat the event list once for a seamless ticker loop.
  let stream = $derived(events.length > 0 ? [...events, ...events] : []);
</script>

<div class="ticker">
  <span class="ticker-label">LIVE</span>
  {#if stream.length === 0}
    <span class="ticker-empty">Waiting for activity…</span>
  {:else}
    <div class="ticker-stream">
      {#each stream as ev, i (i)}
        <span class="ev">
          <span class="dim">{timeOf(ev.at)}</span>
          <span class="tag">{ev.kind}</span>
          {#if ev.label}<span class="val">{ev.label}</span>{/if}
          {#if ev.detail}<span class="ok">·</span><span class="ok">{ev.detail}</span>{/if}
        </span>
      {/each}
    </div>
  {/if}
</div>

<style>
  .ticker {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 7px 22px;
    border-bottom: 1px solid var(--ux-border);
    font-family: var(--ux-mono);
    font-size: 10.5px;
    color: var(--ux-text-faint);
    overflow: hidden;
    letter-spacing: 0.02em;
    position: relative;
    height: 30px;
  }
  .ticker::before,
  .ticker::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    width: 80px;
    pointer-events: none;
    z-index: 2;
  }
  .ticker::before {
    left: 0;
    background: linear-gradient(90deg, var(--ux-bg) 0%, transparent 100%);
  }
  .ticker::after {
    right: 0;
    background: linear-gradient(270deg, var(--ux-bg) 0%, transparent 100%);
  }
  .ticker-label {
    color: var(--ux-text-faint);
    font-weight: 500;
    opacity: 0.7;
    flex-shrink: 0;
  }
  .ticker-empty {
    opacity: 0.5;
  }
  .ticker-stream {
    display: flex;
    gap: 28px;
    white-space: nowrap;
    animation: uxTicker 50s linear infinite;
  }
  .ev {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .ev .dim {
    color: var(--ux-text-faint);
    opacity: 0.7;
  }
  .ev .tag {
    color: var(--ux-text-dim);
  }
  .ev .val {
    color: var(--ux-text);
  }
  .ev .ok {
    color: var(--ux-green);
  }
</style>
