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
    /** "macbook pro" | "mac studio" | "mac mini" | "dgx spark" | "linux" etc. */
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

  // NVIDIA logo SVG path
  const NVIDIA_LOGO_PATH =
    "M0.81 0.429V0.299c0.013 -0.001 0.026 -0.002 0.038 -0.002 0.355 -0.011 0.588 0.306 0.588 0.306S1.186 0.952 0.916 0.952c-0.036 0 -0.071 -0.006 -0.105 -0.017V0.542c0.138 0.017 0.166 0.078 0.249 0.216l0.185 -0.155s-0.135 -0.177 -0.362 -0.177c-0.024 -0.001 -0.048 0.001 -0.072 0.003m0 -0.429v0.194l0.038 -0.002c0.494 -0.017 0.816 0.405 0.816 0.405s-0.37 0.45 -0.754 0.45c-0.034 0 -0.066 -0.003 -0.099 -0.009v0.12c0.027 0.003 0.055 0.006 0.082 0.006 0.358 0 0.618 -0.183 0.869 -0.399 0.042 0.034 0.212 0.114 0.247 0.15 -0.238 0.2 -0.794 0.361 -1.11 0.361 -0.03 0 -0.059 -0.002 -0.088 -0.005v0.169h1.362V0zm0 0.935v0.102c-0.331 -0.059 -0.423 -0.404 -0.423 -0.404s0.159 -0.176 0.423 -0.205v0.112h-0.001C0.671 0.524 0.562 0.654 0.562 0.654s0.062 0.218 0.248 0.282m-0.588 -0.316s0.196 -0.29 0.589 -0.32V0.194C0.376 0.229 0 0.597 0 0.597s0.213 0.616 0.81 0.672v-0.112c-0.438 -0.054 -0.588 -0.538 -0.588 -0.538";

  // Tux penguin logo path (viewBox 0 0 100 100)
  const TUX_LOGO_PATH =
    "M50 5C42 5 36 11 36 18c0 4 2 8 5 10C33 32 25 42 25 56v12c0 4 2 7 5 9l-6 4c-2 1-3 3-3 5v3c0 2 2 4 4 4h10l4-4h22l4 4h10c2 0 4-2 4-4v-3c0-2-1-4-3-5l-6-4c3-2 5-5 5-9V56c0-14-8-24-16-28 3-2 5-6 5-10 0-7-6-13-14-13z";
  const TUX_LOGO_VIEWBOX = "0 0 100 100";

  const wireColor = "rgba(179,179,179,0.8)";
  const strokeWidth = 1.5;

  const modelLower = $derived(deviceType.toLowerCase());
  const isSpark = $derived(
    modelLower.includes("dgx") || modelLower.includes("gx10"),
  );
  const isLinux = $derived(!isSpark && modelLower.startsWith("linux"));
  const isLinuxLaptop = $derived(isLinux && modelLower.includes("laptop"));

  // ── DGX Spark dimensions ──
  const dgxW = $derived(size * 1.55);
  const dgxH = $derived(size * 0.58);
  const dgxX = $derived(cx - dgxW / 2);
  const dgxY = $derived(cy - dgxH / 2);
  const dgxChassisX = $derived(dgxX - dgxW * 0.03);
  const dgxChassisW = $derived(dgxW * 1.05);
  const dgxHandleW = $derived(dgxW * 0.27);
  const dgxHandleGap = $derived(dgxH * 0.05);
  const dgxHandleH = $derived(dgxH - dgxHandleGap * 2);
  const dgxHandleY = $derived(dgxY + dgxHandleGap);
  const dgxInnerHandleW = $derived(dgxW * 0.12);
  const dgxInnerHandleH = $derived(dgxHandleH - dgxH * 0.06);
  const dgxLeftHandleX = $derived(dgxX + 4);
  const dgxRightHandleX = $derived(dgxX + dgxW - dgxHandleW - 4);
  const dgxClipId = $derived(`di-dgx-${uid}`);
  const dgxTextureId = $derived(`di-dgx-tex-${uid}`);

  // ── Linux Desktop dimensions (reuses Mac Studio proportions) ──
  const linuxDesktopClipId = $derived(`di-linux-desktop-${uid}`);

  // ── Linux Laptop dimensions (reuses MacBook proportions) ──
  const linuxScreenClipId = $derived(`di-linux-screen-${uid}`);

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

{#if isSpark}
  <!-- DGX Spark -->
  <defs>
    <clipPath id={dgxClipId}>
      <rect x={dgxX} y={dgxY} width={dgxW} height={dgxH} rx="3" />
    </clipPath>
    <pattern
      id={dgxTextureId}
      patternUnits="userSpaceOnUse"
      width="8"
      height="8"
    >
      <rect width="8" height="8" fill="#6f6248" />
      <circle cx="2" cy="2" r="1" fill="#5a4f3b" opacity="0.5" />
      <circle cx="6" cy="6" r="1" fill="#4a4232" opacity="0.45" />
    </pattern>
  </defs>

  <!-- Main body -->
  <rect
    x={dgxChassisX}
    y={dgxY}
    width={dgxChassisW}
    height={dgxH}
    rx="3"
    fill="url(#{dgxTextureId})"
    stroke={wireColor}
    stroke-width={strokeWidth}
  />

  <!-- Side border accents -->
  <rect
    x={dgxChassisX}
    y={dgxY}
    width={dgxW * 0.02}
    height={dgxH}
    fill="#8a7a56"
  />
  <rect
    x={dgxChassisX + dgxChassisW - dgxW * 0.02}
    y={dgxY}
    width={dgxW * 0.02}
    height={dgxH}
    fill="#8a7a56"
  />

  <!-- Memory fill -->
  {#if ramPercent > 0}
    <rect
      x={dgxX}
      y={dgxY + dgxH - (ramPercent / 100) * dgxH}
      width={dgxW}
      height={(ramPercent / 100) * dgxH}
      fill="rgba(255,215,0,0.45)"
      clip-path="url(#{dgxClipId})"
    />
  {/if}

  <!-- Left handle -->
  <rect
    x={dgxLeftHandleX}
    y={dgxHandleY}
    width={dgxHandleW}
    height={dgxHandleH}
    rx="2.4"
    fill="#b3a170"
    stroke="#403723"
    stroke-width="0.7"
  />
  <rect
    x={dgxLeftHandleX + dgxHandleW * 0.06}
    y={dgxHandleY + dgxH * 0.03}
    width={dgxInnerHandleW}
    height={dgxInnerHandleH}
    rx="1.6"
    fill="#8a7a56"
  />

  <!-- Right handle -->
  <rect
    x={dgxRightHandleX}
    y={dgxHandleY}
    width={dgxHandleW}
    height={dgxHandleH}
    rx="2.4"
    fill="#b3a170"
    stroke="#403723"
    stroke-width="0.7"
  />
  <rect
    x={dgxRightHandleX + dgxHandleW - dgxInnerHandleW - dgxHandleW * 0.08}
    y={dgxHandleY + dgxH * 0.03}
    width={dgxInnerHandleW}
    height={dgxInnerHandleH}
    rx="1.6"
    fill="#8a7a56"
  />

  <!-- NVIDIA logo (rotated 90deg on left handle) -->
  {@const badgeW = dgxW * 0.09}
  {@const badgeH = dgxHandleH * 0.5}
  {@const badgeX = dgxLeftHandleX + dgxHandleW - badgeW - dgxHandleW * 0.06}
  {@const badgeYPos = dgxHandleY + (dgxHandleH - badgeH) / 2}
  {@const textSz = badgeW * 0.58}
  {@const logoW = textSz * 1.2}
  {@const logoH = logoW * (1.438 / 2.174)}
  {@const ctrX = badgeX + badgeW / 2 - badgeW * 0.03}
  {@const ctrY = badgeYPos + badgeH / 2}
  {@const labelGap = badgeW * 0.15}
  {@const totalW = logoW + labelGap + textSz * 3.6}
  <g transform="rotate(90 {ctrX} {ctrY})">
    <svg
      x={ctrX - totalW / 2}
      y={ctrY - logoH / 2}
      width={logoW}
      height={logoH}
      viewBox="0 0 2.174 1.438"
    >
      <path d={NVIDIA_LOGO_PATH} fill="#76b900" />
    </svg>
    <text
      x={ctrX - totalW / 2 + logoW + labelGap}
      y={ctrY}
      text-anchor="start"
      dominant-baseline="middle"
      fill="#8a7a56"
      font-size={textSz}
      font-family="monospace"
      font-weight="700">NVIDIA</text
    >
  </g>
{:else if isLinuxLaptop}
  <!-- Linux Laptop — MacBook shape with Tux logo -->
  <defs>
    <clipPath id={linuxScreenClipId}>
      <rect
        x={mbScreenX + mbBezel}
        y={mbY + mbBezel}
        width={mbScreenW - mbBezel * 2}
        height={mbScreenH - mbBezel * 2}
        rx="2"
      />
    </clipPath>
  </defs>

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
  <rect
    x={mbScreenX + mbBezel}
    y={mbY + mbBezel}
    width={mbScreenW - mbBezel * 2}
    height={mbScreenH - mbBezel * 2}
    rx="2"
    fill="#0a0a12"
  />
  {#if ramPercent > 0}
    <rect
      x={mbScreenX + mbBezel}
      y={mbY + mbBezel + (mbMemTotalH - mbMemH)}
      width={mbScreenW - mbBezel * 2}
      height={mbMemH}
      fill="rgba(255,215,0,0.85)"
      clip-path="url(#{linuxScreenClipId})"
    />
  {/if}

  <!-- Tux logo on screen -->
  {@const tuxH = mbScreenH * 0.35}
  {@const tuxW = tuxH}
  {@const tuxX = cx - tuxW / 2}
  {@const tuxY = mbY + mbScreenH / 2 - tuxH / 2}
  <svg x={tuxX} y={tuxY} width={tuxW} height={tuxH} viewBox={TUX_LOGO_VIEWBOX}>
    <path d={TUX_LOGO_PATH} fill="#FFFFFF" opacity="0.9" />
  </svg>

  <path
    d="M {mbBaseTopX} {mbBaseY} L {mbBaseTopX +
      mbBaseTopW} {mbBaseY} L {mbBaseBottomX + mbBaseBottomW} {mbBaseY +
      mbBaseH} L {mbBaseBottomX} {mbBaseY + mbBaseH} Z"
    fill="#2c2c2c"
    stroke={wireColor}
    stroke-width="1"
  />
  <rect
    x={mbKbX}
    y={mbKbY}
    width={mbKbW}
    height={mbKbH}
    fill="rgba(0,0,0,0.2)"
    rx="2"
  />
  <rect
    x={mbTpX}
    y={mbTpY}
    width={mbTpW}
    height={mbTpH}
    fill="rgba(255,255,255,0.08)"
    rx="2"
  />
{:else if isLinux}
  <!-- Linux Desktop — Mac Studio shape with Tux logo -->
  <defs>
    <clipPath id={linuxDesktopClipId}>
      <rect
        x={studioX}
        y={studioY + studioTopH}
        width={studioW}
        height={studioH - studioTopH}
        rx={studioCorner - 1}
      />
    </clipPath>
  </defs>

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
  {#if ramPercent > 0}
    <rect
      x={studioX}
      y={studioY + studioTopH + (studioMemTotalH - studioMemH)}
      width={studioW}
      height={studioMemH}
      fill="rgba(255,215,0,0.75)"
      clip-path="url(#{linuxDesktopClipId})"
    />
  {/if}

  <!-- Tux logo centered -->
  {@const dtH = (studioH - studioTopH) * 0.55}
  {@const dtW = dtH}
  {@const dtX = cx - dtW / 2}
  {@const dtY = studioY + studioTopH + (studioH - studioTopH) / 2 - dtH / 2}
  <svg x={dtX} y={dtY} width={dtW} height={dtH} viewBox={TUX_LOGO_VIEWBOX}>
    <path d={TUX_LOGO_PATH} fill="rgba(255,255,255,0.6)" />
  </svg>
{:else if modelLower === "mac studio" || modelLower === "mac mini"}
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
