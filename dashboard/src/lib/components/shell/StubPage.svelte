<script lang="ts">
  interface Props {
    eyebrow: string;
    title: string;
    subtitle?: string;
    nextLabel?: string;
    nextHref?: string;
  }
  let { eyebrow, title, subtitle, nextLabel, nextHref = "#/legacy" }: Props = $props();
</script>

<div class="page-header">
  <div>
    <div class="eyebrow">{eyebrow}</div>
    <h1>{title}</h1>
    {#if subtitle}
      <div class="subtitle">{subtitle}</div>
    {/if}
  </div>
</div>

<div class="card empty-card">
  <div class="empty-eyebrow">COMING SOON</div>
  <div class="empty-title">This surface is being rebuilt for the new shell.</div>
  <div class="empty-sub">
    The full feature set is still available in the legacy command-center view.
  </div>
  <a class="btn" href={nextHref}>
    {nextLabel ?? "Open legacy view"} →
  </a>
</div>

<style>
  .page-header {
    margin-bottom: 36px;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ux-text-faint);
    font-size: 10px;
    font-weight: 600;
    font-family: var(--ux-mono);
    margin-bottom: 6px;
  }
  h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ux-text);
  }
  .subtitle {
    color: var(--ux-text-dim);
    font-size: 13px;
    margin-top: 6px;
  }
  .card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
  }
  .empty-card {
    padding: 64px 32px;
    text-align: center;
    background:
      radial-gradient(ellipse 60% 50% at center, var(--ux-accent-bg) 0%, transparent 70%),
      var(--ux-card);
  }
  .empty-eyebrow {
    font-family: var(--ux-mono);
    font-size: 10px;
    color: var(--ux-text-faint);
    letter-spacing: 0.14em;
    margin-bottom: 12px;
  }
  .empty-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--ux-text);
    margin-bottom: 8px;
  }
  .empty-sub {
    font-size: 13px;
    color: var(--ux-text-dim);
    margin-bottom: 24px;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    font-family: var(--ux-sans);
    font-size: 13px;
    font-weight: 500;
    padding: 9px 16px;
    border-radius: var(--ux-radius-sm);
    border: 1px solid var(--ux-border-strong);
    background: var(--ux-bg-raised);
    color: var(--ux-text);
    text-decoration: none;
    transition: background 120ms;
  }
  .btn:hover {
    background: var(--ux-bg-hover);
  }
</style>
