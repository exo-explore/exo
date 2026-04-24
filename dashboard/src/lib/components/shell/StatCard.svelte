<script lang="ts">
  interface Props {
    label: string;
    value: string;
    valueUnit?: string;
    foot?: string;
    delta?: string;
    accent?: boolean;
  }
  let { label, value, valueUnit, foot, delta, accent = false }: Props = $props();
</script>

<div class="stat-card">
  <div class="stat-label">
    <span>{label}</span>
    {#if delta}
      <span class="stat-delta" class:accent>{delta}</span>
    {/if}
  </div>
  <div class="stat-value">
    {value}
    {#if valueUnit}<span class="unit">{valueUnit}</span>{/if}
  </div>
  {#if foot}
    <div class="stat-foot">{foot}</div>
  {/if}
</div>

<style>
  .stat-card {
    background: var(--ux-card);
    border: 1px solid var(--ux-border);
    border-radius: var(--ux-radius);
    padding: 18px 20px 20px;
    position: relative;
    overflow: hidden;
  }
  .stat-label {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ux-text-faint);
    font-size: 10px;
    font-weight: 600;
    font-family: var(--ux-mono);
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .stat-delta {
    text-transform: none;
    letter-spacing: 0;
    color: var(--ux-green);
    font-size: 10px;
  }
  .stat-delta.accent {
    color: var(--ux-accent);
  }
  .stat-value {
    font-family: var(--ux-mono);
    font-size: 34px;
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ux-text);
  }
  .stat-value .unit {
    color: var(--ux-text-faint);
    font-size: 16px;
    margin-left: 4px;
    font-weight: 400;
  }
  .stat-foot {
    margin-top: 12px;
    font-size: 11px;
    color: var(--ux-text-faint);
    font-family: var(--ux-mono);
  }
</style>
