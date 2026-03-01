<script lang="ts">
  /**
   * DeviceIcon — renders a device icon as an SVG <g> element.
   * Uses the exact same proportional math as TopologyGraph.svelte
   * so that devices look identical in both the topology view and
   * the onboarding animation.
   *
   * Must be placed inside an <svg> element.
   */

  interface Props {
    /** "macbook pro" | "mac studio" | "mac mini" etc. */
    deviceType: string;
    /** Center X coordinate in SVG space */
    cx: number;
    /** Center Y coordinate in SVG space */
    cy: number;
    /** Base sizing factor (equivalent to TopologyGraph's nodeRadius) */
    size?: number;
    /** RAM usage 0–100 */
    ramPercent?: number;
    /** Unique id suffix for clip-path ids */
    uid?: string;
  }

  let {
    deviceType,
    cx,
    cy,
    size = 60,
    ramPercent = 60,
    uid = "dev",
  }: Props = $props();

  // Apple logo path — same constant used by TopologyGraph
  const APPLE_LOGO_PATH =
    "M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z";
  const LOGO_NATIVE_WIDTH = 814;
  const LOGO_NATIVE_HEIGHT = 1000;

  const wireColor = "rgba(179,179,179,0.8)";
  const strokeWidth = 1.5;

  const modelLower = $derived(deviceType.toLowerCase());

  // ── Mac Studio dimensions (same ratios as TopologyGraph) ──
  const studioW = $derived(size * 1.25);
  const studioH = $derived(size * 0.85);
  const studioX = $derived(cx - studioW / 2);
  const studioY = $derived(cy - studioH / 2);
  const studioCorner = 4;
  const studioTopH = $derived(studioH * 0.15);

  // Studio front panel details
  const studioSlotH = $derived(studioH * 0.14);
  const studioVSlotW = $derived(studioW * 0.05);
  const studioVSlotY = $derived(
    studioY + studioTopH + (studioH - studioTopH) * 0.6,
  );
  const studioVSlot1X = $derived(studioX + studioW * 0.18);
  const studioVSlot2X = $derived(studioX + studioW * 0.28);
  const studioHSlotW = $derived(studioW * 0.2);
  const studioHSlotX = $derived(studioX + studioW * 0.5 - studioHSlotW / 2);

  // Studio memory fill
  const studioMemTotalH = $derived(studioH - studioTopH);
  const studioMemH = $derived((ramPercent / 100) * studioMemTotalH);

  // ── MacBook dimensions (same ratios as TopologyGraph) ──
  const mbW = $derived((size * 1.6 * 0.85) / 1.15);
  const mbH = $derived(size * 0.85);
  const mbX = $derived(cx - mbW / 2);
  const mbY = $derived(cy - mbH / 2);

  const mbScreenH = $derived(mbH * 0.7);
  const mbBaseH = $derived(mbH * 0.3);
  const mbScreenW = $derived(mbW * 0.85);
  const mbScreenX = $derived(cx - mbScreenW / 2);
  const mbBezel = 3;

  // MacBook memory fill
  const mbMemTotalH = $derived(mbScreenH - mbBezel * 2);
  const mbMemH = $derived((ramPercent / 100) * mbMemTotalH);

  // Apple logo sizing
  const mbLogoTargetH = $derived(mbScreenH * 0.22);
  const mbLogoScale = $derived(mbLogoTargetH / LOGO_NATIVE_HEIGHT);
  const mbLogoX = $derived(cx - (LOGO_NATIVE_WIDTH * mbLogoScale) / 2);
  const mbLogoY = $derived(
    mbY + mbScreenH / 2 - (LOGO_NATIVE_HEIGHT * mbLogoScale) / 2,
  );

  // MacBook base (trapezoidal)
  const mbBaseY = $derived(mbY + mbScreenH);
  const mbBaseTopW = $derived(mbScreenW);
  const mbBaseBottomW = $derived(mbW);
  const mbBaseTopX = $derived(cx - mbBaseTopW / 2);
  const mbBaseBottomX = $derived(cx - mbBaseBottomW / 2);

  // Keyboard
  const mbKbX = $derived(mbBaseTopX + 6);
  const mbKbY = $derived(mbBaseY + 3);
  const mbKbW = $derived(mbBaseTopW - 12);
  const mbKbH = $derived(mbBaseH * 0.55);

  // Trackpad
  const mbTpW = $derived(mbBaseTopW * 0.4);
  const mbTpX = $derived(cx - mbTpW / 2);
  const mbTpY = $derived(mbBaseY + mbKbH + 5);
  const mbTpH = $derived(mbBaseH * 0.3);

  // Clip IDs
  const screenClipId = $derived(`di-screen-${uid}`);
  const studioClipId = $derived(`di-studio-${uid}`);
</script>

{#if modelLower === "mac studio" || modelLower === "mac mini"}
  <!-- Mac Studio / Mac Mini -->
  <defs>
    <clipPath id={studioClipId}>
      <rect
        x={studioX}
        y={studioY + studioTopH}
        width={studioW}
        height={studioH - studioTopH}
        rx={studioCorner - 1}
      />
    </clipPath>
  </defs>

  <!-- Main body -->
  <rect
    x={studioX}
    y={studioY}
    width={studioW}
    height={studioH}
    rx={studioCorner}
    fill="#1a1a1a"
    stroke={wireColor}
    stroke-width={strokeWidth}
  />

  <!-- Memory fill -->
  {#if ramPercent > 0}
    <rect
      x={studioX}
      y={studioY + studioTopH + (studioMemTotalH - studioMemH)}
      width={studioW}
      height={studioMemH}
      fill="rgba(255,215,0,0.75)"
      clip-path="url(#{studioClipId})"
    />
  {/if}

  <!-- Top surface divider -->
  <line
    x1={studioX}
    y1={studioY + studioTopH}
    x2={studioX + studioW}
    y2={studioY + studioTopH}
    stroke="rgba(179,179,179,0.3)"
    stroke-width="0.5"
  />

  <!-- Front panel: vertical slots -->
  <rect
    x={studioVSlot1X - studioVSlotW / 2}
    y={studioVSlotY}
    width={studioVSlotW}
    height={studioSlotH}
    fill="rgba(0,0,0,0.35)"
    rx="1.5"
  />
  <rect
    x={studioVSlot2X - studioVSlotW / 2}
    y={studioVSlotY}
    width={studioVSlotW}
    height={studioSlotH}
    fill="rgba(0,0,0,0.35)"
    rx="1.5"
  />

  <!-- Horizontal slot (SD card) -->
  <rect
    x={studioHSlotX}
    y={studioVSlotY}
    width={studioHSlotW}
    height={studioSlotH * 0.6}
    fill="rgba(0,0,0,0.35)"
    rx="1"
  />
{:else}
  <!-- MacBook Pro -->
  <defs>
    <clipPath id={screenClipId}>
      <rect
        x={mbScreenX + mbBezel}
        y={mbY + mbBezel}
        width={mbScreenW - mbBezel * 2}
        height={mbScreenH - mbBezel * 2}
        rx="2"
      />
    </clipPath>
  </defs>

  <!-- Screen outer frame -->
  <rect
    x={mbScreenX}
    y={mbY}
    width={mbScreenW}
    height={mbScreenH}
    rx="3"
    fill="#1a1a1a"
    stroke={wireColor}
    stroke-width={strokeWidth}
  />

  <!-- Screen inner (dark) -->
  <rect
    x={mbScreenX + mbBezel}
    y={mbY + mbBezel}
    width={mbScreenW - mbBezel * 2}
    height={mbScreenH - mbBezel * 2}
    rx="2"
    fill="#0a0a12"
  />

  <!-- Memory fill on screen -->
  {#if ramPercent > 0}
    <rect
      x={mbScreenX + mbBezel}
      y={mbY + mbBezel + (mbMemTotalH - mbMemH)}
      width={mbScreenW - mbBezel * 2}
      height={mbMemH}
      fill="rgba(255,215,0,0.85)"
      clip-path="url(#{screenClipId})"
    />
  {/if}

  <!-- Apple logo -->
  <path
    d={APPLE_LOGO_PATH}
    transform="translate({mbLogoX}, {mbLogoY}) scale({mbLogoScale})"
    fill="#FFFFFF"
    opacity="0.9"
  />

  <!-- Keyboard base (trapezoidal) -->
  <path
    d="M {mbBaseTopX} {mbBaseY} L {mbBaseTopX +
      mbBaseTopW} {mbBaseY} L {mbBaseBottomX + mbBaseBottomW} {mbBaseY +
      mbBaseH} L {mbBaseBottomX} {mbBaseY + mbBaseH} Z"
    fill="#2c2c2c"
    stroke={wireColor}
    stroke-width="1"
  />

  <!-- Keyboard area -->
  <rect
    x={mbKbX}
    y={mbKbY}
    width={mbKbW}
    height={mbKbH}
    fill="rgba(0,0,0,0.2)"
    rx="2"
  />

  <!-- Trackpad -->
  <rect
    x={mbTpX}
    y={mbTpY}
    width={mbTpW}
    height={mbTpH}
    fill="rgba(255,255,255,0.08)"
    rx="2"
  />
{/if}
