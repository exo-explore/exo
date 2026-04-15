<script lang="ts">
  interface Point {
    t: number;
    v: number | null;
  }
  interface Props {
    title: string;
    unit: string;
    points: Point[];
    color?: string;
    formatY?: (v: number) => string;
  }

  let {
    title,
    unit,
    points,
    color = "#facc15",
    formatY = (v: number) => v.toFixed(0),
  }: Props = $props();

  const width = 520;
  const height = 120;
  const padLeft = 44;
  const padRight = 12;
  const padTop = 18;
  const padBottom = 22;

  const plot = $derived.by(() => {
    const valid = points.filter((p) => p.v !== null) as {
      t: number;
      v: number;
    }[];
    if (valid.length === 0) {
      return null;
    }
    const minT = Math.min(...valid.map((p) => p.t));
    const maxT = Math.max(...valid.map((p) => p.t));
    const minV = 0;
    const maxV = Math.max(...valid.map((p) => p.v), 1);
    const xSpan = maxT - minT || 1;
    const ySpan = maxV - minV || 1;
    const innerW = width - padLeft - padRight;
    const innerH = height - padTop - padBottom;
    const xs = (t: number) => padLeft + ((t - minT) / xSpan) * innerW;
    const ys = (v: number) => padTop + innerH - ((v - minV) / ySpan) * innerH;
    const pathD = valid
      .map((p, i) => `${i === 0 ? "M" : "L"}${xs(p.t)},${ys(p.v)}`)
      .join(" ");
    const circles = valid.map((p) => ({ cx: xs(p.t), cy: ys(p.v), v: p.v }));
    const yTicks = [0, maxV / 2, maxV].map((v) => ({ y: ys(v), v }));
    return { pathD, circles, yTicks, count: valid.length, maxV };
  });
</script>

<div
  class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-3 space-y-2"
>
  <div class="flex items-baseline justify-between gap-2">
    <div class="text-xs font-mono text-exo-light-gray uppercase tracking-wider">
      {title}
    </div>
    <div class="text-[10px] font-mono text-exo-light-gray/70">
      {plot ? `${plot.count} points` : "no data"}
    </div>
  </div>
  {#if plot}
    <svg
      viewBox="0 0 {width} {height}"
      class="w-full h-auto"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label={title}
    >
      {#each plot.yTicks as tick}
        <line
          x1={padLeft}
          x2={width - padRight}
          y1={tick.y}
          y2={tick.y}
          stroke="rgba(255,255,255,0.07)"
          stroke-width="1"
        />
        <text
          x={padLeft - 4}
          y={tick.y + 3}
          text-anchor="end"
          font-family="ui-monospace, monospace"
          font-size="9"
          fill="rgba(255,255,255,0.5)"
        >
          {formatY(tick.v)}
        </text>
      {/each}
      <path d={plot.pathD} fill="none" stroke={color} stroke-width="1.5" />
      {#each plot.circles as c}
        <circle cx={c.cx} cy={c.cy} r="2.5" fill={color}>
          <title>{formatY(c.v)} {unit}</title>
        </circle>
      {/each}
    </svg>
  {:else}
    <div class="text-xs font-mono text-exo-light-gray/50 text-center py-6">
      no data
    </div>
  {/if}
</div>
