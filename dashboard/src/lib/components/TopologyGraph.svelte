<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as d3 from "d3";
  import {
    topologyData,
    isTopologyMinimized,
    debugMode,
    nodeThunderboltBridge,
    nodeRdmaCtl,
    nodeIdentities,
    type NodeInfo,
  } from "$lib/stores/app.svelte";

  interface Props {
    class?: string;
    highlightedNodes?: Set<string>;
    filteredNodes?: Set<string>;
    onNodeClick?: (nodeId: string) => void;
  }

  let {
    class: className = "",
    highlightedNodes = new Set(),
    filteredNodes = new Set(),
    onNodeClick,
  }: Props = $props();

  let svgContainer: SVGSVGElement | undefined = $state();
  let resizeObserver: ResizeObserver | undefined;
  let hoveredNodeId = $state<string | null>(null);

  const isMinimized = $derived(isTopologyMinimized());
  const data = $derived(topologyData());
  const debugEnabled = $derived(debugMode());
  const tbBridgeData = $derived(nodeThunderboltBridge());
  const rdmaCtlData = $derived(nodeRdmaCtl());
  const identitiesData = $derived(nodeIdentities());

  function getNodeLabel(nodeId: string): string {
    const node = data?.nodes?.[nodeId];
    return node?.friendly_name || nodeId.slice(0, 8);
  }

  function getInterfaceLabel(
    nodeId: string,
    ip?: string,
  ): { label: string; missing: boolean } {
    if (!ip) return { label: "?", missing: true };

    // Strip port if present (e.g., "192.168.1.1:8080" -> "192.168.1.1")
    const cleanIp =
      ip.includes(":") && !ip.includes("[") ? ip.split(":")[0] : ip;

    // Helper to check a node's interfaces
    function checkNode(node: NodeInfo | undefined): string | null {
      if (!node) return null;

      const matchFromInterfaces = node.network_interfaces?.find((iface) =>
        (iface.addresses || []).some((addr) => addr === cleanIp || addr === ip),
      );
      if (matchFromInterfaces?.name) {
        return matchFromInterfaces.name;
      }

      if (node.ip_to_interface) {
        const mapped =
          node.ip_to_interface[cleanIp] ||
          (ip ? node.ip_to_interface[ip] : undefined);
        if (mapped && mapped.trim().length > 0) {
          return mapped;
        }
      }
      return null;
    }

    // Try specified node first
    const result = checkNode(data?.nodes?.[nodeId]);
    if (result) return { label: result, missing: false };

    // Fallback: search all nodes for this IP
    for (const [, otherNode] of Object.entries(data?.nodes || {})) {
      const otherResult = checkNode(otherNode);
      if (otherResult) return { label: otherResult, missing: false };
    }

    return { label: "?", missing: true };
  }

  function wrapLine(text: string, maxLen: number): string[] {
    if (text.length <= maxLen) return [text];
    const words = text.split(" ");
    const lines: string[] = [];
    let current = "";
    for (const word of words) {
      if (word.length > maxLen) {
        if (current) {
          lines.push(current);
          current = "";
        }
        for (let i = 0; i < word.length; i += maxLen) {
          lines.push(word.slice(i, i + maxLen));
        }
      } else if ((current + " " + word).trim().length > maxLen) {
        lines.push(current);
        current = word;
      } else {
        current = current ? `${current} ${word}` : word;
      }
    }
    if (current) lines.push(current);
    return lines;
  }

  // Apple logo path for MacBook Pro screen
  const APPLE_LOGO_PATH =
    "M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z";
  const LOGO_NATIVE_WIDTH = 814;
  const LOGO_NATIVE_HEIGHT = 1000;

  // NVIDIA logo SVG path (from exo-nvidia)
  const NVIDIA_LOGO_PATH =
    "M0.81 0.429V0.299c0.013 -0.001 0.026 -0.002 0.038 -0.002 0.355 -0.011 0.588 0.306 0.588 0.306S1.186 0.952 0.916 0.952c-0.036 0 -0.071 -0.006 -0.105 -0.017V0.542c0.138 0.017 0.166 0.078 0.249 0.216l0.185 -0.155s-0.135 -0.177 -0.362 -0.177c-0.024 -0.001 -0.048 0.001 -0.072 0.003m0 -0.429v0.194l0.038 -0.002c0.494 -0.017 0.816 0.405 0.816 0.405s-0.37 0.45 -0.754 0.45c-0.034 0 -0.066 -0.003 -0.099 -0.009v0.12c0.027 0.003 0.055 0.006 0.082 0.006 0.358 0 0.618 -0.183 0.869 -0.399 0.042 0.034 0.212 0.114 0.247 0.15 -0.238 0.2 -0.794 0.361 -1.11 0.361 -0.03 0 -0.059 -0.002 -0.088 -0.005v0.169h1.362V0zm0 0.935v0.102c-0.331 -0.059 -0.423 -0.404 -0.423 -0.404s0.159 -0.176 0.423 -0.205v0.112h-0.001C0.671 0.524 0.562 0.654 0.562 0.654s0.062 0.218 0.248 0.282m-0.588 -0.316s0.196 -0.29 0.589 -0.32V0.194C0.376 0.229 0 0.597 0 0.597s0.213 0.616 0.81 0.672v-0.112c-0.438 -0.054 -0.588 -0.538 -0.588 -0.538";

  // Tux penguin SVG path — simplified silhouette for topology view
  // viewBox: 0 0 100 100
  const TUX_BODY_PATH =
    "M50 8c-8 0-14 6-14 13 0 4 2 8 5 10-8 4-16 14-16 28v12c0 4 2 7 5 9l-6 4c-2 1-3 3-3 5v3c0 2 2 4 4 4h10l4-4h22l4 4h10c2 0 4-2 4-4v-3c0-2-1-4-3-5l-6-4c3-2 5-5 5-9V59c0-14-8-24-16-28 3-2 5-6 5-10 0-7-6-13-14-13z";
  const TUX_BELLY_PATH =
    "M38 52c0-8 5-15 12-15s12 7 12 15v14c0 4-5 7-12 7s-12-3-12-7V52z";

  function formatBytes(bytes: number, decimals = 1): string {
    if (!bytes || bytes === 0) return "0B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + sizes[i];
  }

  function getTemperatureColor(temp: number): string {
    // Default for N/A temp - light gray
    if (isNaN(temp) || temp === null) return "rgba(179, 179, 179, 0.8)";

    const coolTemp = 45; // Temp for pure blue
    const midTemp = 57.5; // Temp for pure yellow
    const hotTemp = 75; // Temp for pure red

    const coolColor = { r: 93, g: 173, b: 226 }; // #5DADE2 (Blue)
    const midColor = { r: 255, g: 215, b: 0 }; // #FFD700 (Yellow)
    const hotColor = { r: 244, g: 67, b: 54 }; // #F44336 (Red)

    let r: number, g: number, b: number;

    if (temp <= coolTemp) {
      ({ r, g, b } = coolColor);
    } else if (temp <= midTemp) {
      const ratio = (temp - coolTemp) / (midTemp - coolTemp);
      r = Math.round(coolColor.r * (1 - ratio) + midColor.r * ratio);
      g = Math.round(coolColor.g * (1 - ratio) + midColor.g * ratio);
      b = Math.round(coolColor.b * (1 - ratio) + midColor.b * ratio);
    } else if (temp < hotTemp) {
      const ratio = (temp - midTemp) / (hotTemp - midTemp);
      r = Math.round(midColor.r * (1 - ratio) + hotColor.r * ratio);
      g = Math.round(midColor.g * (1 - ratio) + hotColor.g * ratio);
      b = Math.round(midColor.b * (1 - ratio) + hotColor.b * ratio);
    } else {
      ({ r, g, b } = hotColor);
    }

    return `rgb(${r}, ${g}, ${b})`;
  }

  function renderGraph() {
    if (!svgContainer || !data) return;

    d3.select(svgContainer).selectAll("*").remove();

    const nodes = data.nodes || {};
    const edges = data.edges || [];
    const nodeIds = Object.keys(nodes);

    const rect = svgContainer.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const centerX = width / 2;
    const centerY = height / 2;

    const svg = d3.select(svgContainer);

    // Add defs for clip paths and filters
    const defs = svg.append("defs");

    // Glow filter
    const glowFilter = defs
      .append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    glowFilter
      .append("feGaussianBlur")
      .attr("stdDeviation", "2")
      .attr("result", "coloredBlur");
    const glowMerge = glowFilter.append("feMerge");
    glowMerge.append("feMergeNode").attr("in", "coloredBlur");
    glowMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrowhead marker for directional edges
    const marker = defs
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", "10")
      .attr("refY", "5")
      .attr("markerWidth", "11")
      .attr("markerHeight", "11")
      .attr("orient", "auto-start-reverse");
    marker
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10")
      .attr("fill", "none")
      .attr("stroke", "var(--exo-light-gray, #B3B3B3)")
      .attr("stroke-width", "1.6")
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .style("animation", "none");

    if (nodeIds.length === 0) {
      svg
        .append("text")
        .attr("x", centerX)
        .attr("y", centerY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("fill", "rgba(255,215,0,0.4)")
        .attr("font-size", isMinimized ? 10 : 12)
        .attr("font-family", "SF Mono, monospace")
        .attr("letter-spacing", "0.1em")
        .text("AWAITING NODES");
      return;
    }

    const numNodes = nodeIds.length;
    const minDimension = Math.min(width, height);

    // Dynamic scaling - larger nodes for big displays
    const sizeScale =
      numNodes === 1 ? 1 : Math.max(0.6, 1 - (numNodes - 1) * 0.1);
    const baseNodeRadius = isMinimized
      ? Math.max(36, Math.min(60, minDimension * 0.22))
      : Math.min(120, minDimension * 0.2);
    const nodeRadius = baseNodeRadius * sizeScale;

    // Orbit radius - balanced spacing for nodes
    const circumference = numNodes * nodeRadius * 4;
    const radiusFromCircumference = circumference / (2 * Math.PI);
    const minOrbitRadius = Math.max(
      radiusFromCircumference,
      minDimension * 0.18,
    );
    const maxOrbitRadius = minDimension * 0.3;
    const orbitRadius = isMinimized
      ? Math.min(maxOrbitRadius, Math.max(minOrbitRadius, minDimension * 0.26))
      : Math.min(
          maxOrbitRadius,
          Math.max(minOrbitRadius, minDimension * (0.22 + numNodes * 0.02)),
        );

    // Determine display mode based on space and node count
    const showFullLabels = !isMinimized && numNodes <= 4;
    const showCompactLabels = !isMinimized && numNodes > 4;

    // Add padding for labels (top/bottom)
    const topPadding = 70; // Space for "NETWORK TOPOLOGY" label and node names
    const bottomPadding = 70; // Space for stats and bottom label
    const safeCenterY = topPadding + (height - topPadding - bottomPadding) / 2;

    // Calculate node positions
    const nodesWithPositions = nodeIds.map((id, index) => {
      if (numNodes === 1) {
        // Single node: center it
        return {
          id,
          data: nodes[id],
          x: centerX,
          y: safeCenterY,
        };
      }
      // Distribute nodes around the orbit
      // Start from top (-90 degrees) and go clockwise
      const angle = (index / numNodes) * 2 * Math.PI - Math.PI / 2;
      return {
        id,
        data: nodes[id],
        x: centerX + orbitRadius * Math.cos(angle),
        y: safeCenterY + orbitRadius * Math.sin(angle),
      };
    });

    const positionById: Record<string, { x: number; y: number }> = {};
    nodesWithPositions.forEach((n) => {
      positionById[n.id] = { x: n.x, y: n.y };
    });

    // Draw edges
    const linksGroup = svg.append("g").attr("class", "links-group");
    const arrowsGroup = svg.append("g").attr("class", "arrows-group");
    const debugLabelsGroup = svg.append("g").attr("class", "debug-edge-labels");

    type ConnectionInfo = {
      from: string;
      to: string;
      ip: string;
      ifaceLabel: string;
      missingIface: boolean;
    };
    type PairEntry = {
      a: string;
      b: string;
      aToB: boolean;
      bToA: boolean;
      connections: ConnectionInfo[];
    };
    type DebugEdgeLabelEntry = {
      connections: ConnectionInfo[];
      isLeft: boolean;
      isTop: boolean;
      mx: number;
      my: number;
    };
    const pairMap = new Map<string, PairEntry>();
    const debugEdgeLabels: DebugEdgeLabelEntry[] = [];
    edges.forEach((edge) => {
      if (!edge.source || !edge.target || edge.source === edge.target) return;
      if (!positionById[edge.source] || !positionById[edge.target]) return;

      const a = edge.source < edge.target ? edge.source : edge.target;
      const b = edge.source < edge.target ? edge.target : edge.source;
      const key = `${a}|${b}`;
      const entry = pairMap.get(key) || {
        a,
        b,
        aToB: false,
        bToA: false,
        connections: [],
      };

      if (edge.source === a) entry.aToB = true;
      else entry.bToA = true;

      let ip: string;
      let ifaceLabel: string;
      let missingIface: boolean;

      if (edge.sourceRdmaIface || edge.sinkRdmaIface) {
        ip = "RDMA";
        ifaceLabel = `${edge.sourceRdmaIface || "?"} \u2192 ${edge.sinkRdmaIface || "?"}`;
        missingIface = false;
      } else {
        ip = edge.sendBackIp || "?";
        const ifaceInfo = getInterfaceLabel(edge.source, ip);
        ifaceLabel = ifaceInfo.label;
        missingIface = ifaceInfo.missing;
      }

      entry.connections.push({
        from: edge.source,
        to: edge.target,
        ip,
        ifaceLabel,
        missingIface,
      });
      pairMap.set(key, entry);
    });

    pairMap.forEach((entry) => {
      const posA = positionById[entry.a];
      const posB = positionById[entry.b];
      if (!posA || !posB) return;

      // Base dashed line
      linksGroup
        .append("line")
        .attr("x1", posA.x)
        .attr("y1", posA.y)
        .attr("x2", posB.x)
        .attr("y2", posB.y)
        .attr("class", "graph-link");

      // Calculate midpoint and direction for arrows
      const dx = posB.x - posA.x;
      const dy = posB.y - posA.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      const mx = (posA.x + posB.x) / 2;
      const my = (posA.y + posB.y) / 2;
      const tipOffset = 16; // Distance from center for arrow tips
      const carrier = 2; // Short segment length for arrow orientation

      // Arrow A -> B (if connection exists in that direction)
      if (entry.aToB) {
        const tipX = mx - ux * tipOffset;
        const tipY = my - uy * tipOffset;
        arrowsGroup
          .append("line")
          .attr("x1", tipX - ux * carrier)
          .attr("y1", tipY - uy * carrier)
          .attr("x2", tipX)
          .attr("y2", tipY)
          .attr("stroke", "none")
          .attr("fill", "none")
          .attr("marker-end", "url(#arrowhead)");
      }

      // Arrow B -> A (if connection exists in that direction)
      if (entry.bToA) {
        const tipX = mx + ux * tipOffset;
        const tipY = my + uy * tipOffset;
        arrowsGroup
          .append("line")
          .attr("x1", tipX + ux * carrier)
          .attr("y1", tipY + uy * carrier)
          .attr("x2", tipX)
          .attr("y2", tipY)
          .attr("stroke", "none")
          .attr("fill", "none")
          .attr("marker-end", "url(#arrowhead)");
      }

      // Collect debug labels for later positioning at edges
      if (debugEnabled && entry.connections.length > 0) {
        // Determine which side of viewport based on edge midpoint
        const isLeft = mx < centerX;
        const isTop = my < safeCenterY;

        // Store for batch rendering after all edges processed
        debugEdgeLabels.push({
          connections: entry.connections,
          isLeft,
          isTop,
          mx,
          my,
        });
      }
    });

    // Render debug labels at viewport edges/corners
    if (debugEdgeLabels && debugEdgeLabels.length > 0) {
      const fontSize = isMinimized ? 10 : 12;
      const lineHeight = fontSize + 4;
      const padding = 10;

      // Helper to get arrow based on direction vector
      function getArrow(fromId: string, toId: string): string {
        const fromPos = positionById[fromId];
        const toPos = positionById[toId];
        if (!fromPos || !toPos) return "→";

        const dirX = toPos.x - fromPos.x;
        const dirY = toPos.y - fromPos.y;
        const absX = Math.abs(dirX);
        const absY = Math.abs(dirY);

        if (absX > absY * 2) {
          return dirX > 0 ? "→" : "←";
        } else if (absY > absX * 2) {
          return dirY > 0 ? "↓" : "↑";
        } else {
          if (dirX > 0 && dirY > 0) return "↘";
          if (dirX > 0 && dirY < 0) return "↗";
          if (dirX < 0 && dirY > 0) return "↙";
          return "↖";
        }
      }

      // Group by quadrant: topLeft, topRight, bottomLeft, bottomRight
      const quadrants: Record<string, DebugEdgeLabelEntry[]> = {
        topLeft: [],
        topRight: [],
        bottomLeft: [],
        bottomRight: [],
      };

      debugEdgeLabels.forEach((edge) => {
        const key =
          (edge.isTop ? "top" : "bottom") + (edge.isLeft ? "Left" : "Right");
        quadrants[key].push(edge);
      });

      // Render each quadrant
      Object.entries(quadrants).forEach(([quadrant, quadrantEdges]) => {
        if (quadrantEdges.length === 0) return;

        const isLeft = quadrant.includes("Left");
        const isTop = quadrant.includes("top");

        let baseX = isLeft ? padding : width - padding;
        let baseY = isTop ? padding : height - padding;
        const textAnchor = isLeft ? "start" : "end";

        let currentY = baseY;

        quadrantEdges.forEach((edge) => {
          edge.connections.forEach((conn) => {
            const arrow = getArrow(conn.from, conn.to);
            const label = `${arrow} ${conn.ip} ${conn.ifaceLabel}`;
            debugLabelsGroup
              .append("text")
              .attr("x", baseX)
              .attr("y", currentY)
              .attr("text-anchor", textAnchor)
              .attr("dominant-baseline", isTop ? "hanging" : "auto")
              .attr("font-size", fontSize)
              .attr("font-family", "SF Mono, monospace")
              .attr(
                "fill",
                conn.missingIface
                  ? "rgba(248,113,113,0.9)"
                  : "rgba(255,255,255,0.85)",
              )
              .text(label);
            currentY += isTop ? lineHeight : -lineHeight;
          });
        });
      });
    }

    // Draw nodes
    const nodesGroup = svg.append("g").attr("class", "nodes-group");

    nodesWithPositions.forEach((nodeInfo) => {
      const node = nodeInfo.data;
      const macmon = node.macmon_info;
      const modelId = node.system_info?.model_id || "Unknown";
      const friendlyName = node.friendly_name || modelId;

      let ramUsagePercent = 0;
      let gpuTemp = NaN;
      let ramTotal = 0;
      let ramUsed = 0;
      let gpuUsagePercent = 0;
      let sysPower: number | null = null;

      if (macmon) {
        if (macmon.memory && macmon.memory.ram_total > 0) {
          ramUsagePercent =
            (macmon.memory.ram_usage / macmon.memory.ram_total) * 100;
          ramTotal = macmon.memory.ram_total;
          ramUsed = macmon.memory.ram_usage;
        }
        if (macmon.temp && typeof macmon.temp.gpu_temp_avg === "number") {
          gpuTemp = Math.max(30, macmon.temp.gpu_temp_avg);
        }
        if (macmon.gpu_usage) {
          gpuUsagePercent = macmon.gpu_usage[1] * 100;
        }
        if (macmon.sys_power) {
          sysPower = macmon.sys_power;
        }
      }

      let iconBaseWidth = nodeRadius * 1.2;
      let iconBaseHeight = nodeRadius * 1.0;
      const clipPathId = `clip-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;

      const modelLower = modelId.toLowerCase();
      const identity = identitiesData[nodeInfo.id];
      const nameLower = (friendlyName || "").toLowerCase();
      const isSpark =
        modelLower.includes("dgx") || modelLower.includes("gx10");
      const isLinux =
        !isSpark && (modelLower === "linux" || identity?.osVersion === "Linux");

      // Check node states for styling
      const isHighlighted = highlightedNodes.has(nodeInfo.id);
      const isInFilter =
        filteredNodes.size > 0 && filteredNodes.has(nodeInfo.id);
      const isFilteredOut =
        filteredNodes.size > 0 && !filteredNodes.has(nodeInfo.id);
      const isHovered = hoveredNodeId === nodeInfo.id && !isInFilter;

      // Holographic wireframe colors - bright yellow for filter, subtle yellow for hover, grey for filtered out
      const wireColor = isInFilter
        ? "rgba(255,215,0,1)" // Bright yellow for filter selection
        : isHovered
          ? "rgba(255,215,0,0.7)" // Subtle yellow for hover
          : isHighlighted
            ? "rgba(255,215,0,0.9)" // Yellow for instance highlight
            : isFilteredOut
              ? "rgba(140,140,140,0.6)" // Grey for filtered out
              : "rgba(179,179,179,0.8)"; // Default
      const wireColorBright = "rgba(255,255,255,0.9)";
      const fillColor = isInFilter
        ? "rgba(255,215,0,0.25)"
        : isHovered
          ? "rgba(255,215,0,0.12)"
          : isHighlighted
            ? "rgba(255,215,0,0.15)"
            : "rgba(255,215,0,0.08)";
      const strokeWidth = isInFilter
        ? 3
        : isHovered
          ? 2
          : isHighlighted
            ? 2.5
            : 1.5;
      const screenFill = "rgba(0,20,40,0.9)";
      const glowColor = "rgba(255,215,0,0.3)";

      const nodeG = nodesGroup
        .append("g")
        .attr("class", "graph-node")
        .style("cursor", onNodeClick ? "pointer" : "default")
        .style("opacity", isFilteredOut ? 0.5 : 1);

      // Add click and hover handlers - hover just updates state, styling is applied during render
      nodeG
        .on("click", (event: MouseEvent) => {
          if (onNodeClick) {
            event.stopPropagation();
            onNodeClick(nodeInfo.id);
          }
        })
        .on("mouseenter", () => {
          if (onNodeClick) {
            hoveredNodeId = nodeInfo.id;
          }
        })
        .on("mouseleave", () => {
          if (hoveredNodeId === nodeInfo.id) {
            hoveredNodeId = null;
          }
        });

      // Add tooltip
      nodeG
        .append("title")
        .text(
          `${friendlyName}\nID: ${nodeInfo.id.slice(-8)}\nMemory: ${formatBytes(ramUsed)}/${formatBytes(ramTotal)}`,
        );

      if (isSpark) {
        // NVIDIA DGX Spark — gold chassis with textured front, side handles, and NVIDIA badge
        iconBaseWidth = nodeRadius * 1.55;
        iconBaseHeight = nodeRadius * 0.58;
        const x = nodeInfo.x - iconBaseWidth / 2;
        const y = nodeInfo.y - iconBaseHeight / 2;
        const chassisX = x - iconBaseWidth * 0.03;
        const chassisWidth = iconBaseWidth * 1.05;
        const cornerRadius = 3;

        const dgxClipId = `dgx-clip-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
        defs
          .append("clipPath")
          .attr("id", dgxClipId)
          .append("rect")
          .attr("x", x)
          .attr("y", y)
          .attr("width", iconBaseWidth)
          .attr("height", iconBaseHeight)
          .attr("rx", cornerRadius);

        // Chassis texture pattern
        const textureId = `chassis-texture-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
        defs
          .append("pattern")
          .attr("id", textureId)
          .attr("patternUnits", "userSpaceOnUse")
          .attr("width", 8)
          .attr("height", 8);
        const texturePattern = defs.select(`#${textureId}`);
        texturePattern
          .append("rect")
          .attr("width", 8)
          .attr("height", 8)
          .attr("fill", "#6f6248");
        texturePattern
          .append("circle")
          .attr("cx", 2)
          .attr("cy", 2)
          .attr("r", 1)
          .attr("fill", "#5a4f3b")
          .attr("opacity", 0.5);
        texturePattern
          .append("circle")
          .attr("cx", 6)
          .attr("cy", 6)
          .attr("r", 1)
          .attr("fill", "#4a4232")
          .attr("opacity", 0.45);

        // Main body
        nodeG
          .append("rect")
          .attr("class", "node-outline")
          .attr("x", chassisX)
          .attr("y", y)
          .attr("width", chassisWidth)
          .attr("height", iconBaseHeight)
          .attr("rx", cornerRadius)
          .attr("fill", `url(#${textureId})`)
          .attr("stroke", wireColor)
          .attr("stroke-width", strokeWidth);

        // Side border accents
        const sideThickness = iconBaseWidth * 0.02;
        nodeG
          .append("rect")
          .attr("x", chassisX)
          .attr("y", y)
          .attr("width", sideThickness)
          .attr("height", iconBaseHeight)
          .attr("fill", "#8a7a56");
        nodeG
          .append("rect")
          .attr("x", chassisX + chassisWidth - sideThickness)
          .attr("y", y)
          .attr("width", sideThickness)
          .attr("height", iconBaseHeight)
          .attr("fill", "#8a7a56");

        // Memory fill (bottom up)
        if (ramUsagePercent > 0) {
          const memFillHeight = (ramUsagePercent / 100) * iconBaseHeight;
          nodeG
            .append("rect")
            .attr("x", x)
            .attr("y", y + iconBaseHeight - memFillHeight)
            .attr("width", iconBaseWidth)
            .attr("height", memFillHeight)
            .attr("fill", "rgba(255,215,0,0.45)")
            .attr("clip-path", `url(#${dgxClipId})`);
        }

        // Side handles with inner recess
        const handleWidth = iconBaseWidth * 0.27;
        const handleGap = iconBaseHeight * 0.05;
        const handleHeight = iconBaseHeight - handleGap * 2;
        const handleY = y + handleGap;
        const innerHandleWidth = iconBaseWidth * 0.12;
        const innerHandleHeight = handleHeight - iconBaseHeight * 0.06;
        const leftHandleX = x + 4;
        const rightHandleX = x + iconBaseWidth - handleWidth - 4;

        // Left handle
        nodeG
          .append("rect")
          .attr("x", leftHandleX)
          .attr("y", handleY)
          .attr("width", handleWidth)
          .attr("height", handleHeight)
          .attr("rx", 2.4)
          .attr("fill", "#b3a170")
          .attr("stroke", "#403723")
          .attr("stroke-width", 0.7);
        nodeG
          .append("rect")
          .attr("x", leftHandleX + handleWidth * 0.06)
          .attr("y", handleY + iconBaseHeight * 0.03)
          .attr("width", innerHandleWidth)
          .attr("height", innerHandleHeight)
          .attr("rx", 1.6)
          .attr("fill", "#8a7a56");

        // Right handle
        nodeG
          .append("rect")
          .attr("x", rightHandleX)
          .attr("y", handleY)
          .attr("width", handleWidth)
          .attr("height", handleHeight)
          .attr("rx", 2.4)
          .attr("fill", "#b3a170")
          .attr("stroke", "#403723")
          .attr("stroke-width", 0.7);
        nodeG
          .append("rect")
          .attr(
            "x",
            rightHandleX + handleWidth - innerHandleWidth - handleWidth * 0.08,
          )
          .attr("y", handleY + iconBaseHeight * 0.03)
          .attr("width", innerHandleWidth)
          .attr("height", innerHandleHeight)
          .attr("rx", 1.6)
          .attr("fill", "#8a7a56");

        // NVIDIA logo + text label (rotated 90 deg on left handle)
        const badgeWidth = iconBaseWidth * 0.09;
        const badgeHeight = handleHeight * 0.5;
        const badgeX =
          leftHandleX + handleWidth - badgeWidth - handleWidth * 0.06;
        const badgeY = handleY + (handleHeight - badgeHeight) / 2;
        const textSize = badgeWidth * 0.58;
        const logoWidth = textSize * 1.2;
        const logoHeight = logoWidth * (1.438 / 2.174);
        const centerX = badgeX + badgeWidth / 2 - badgeWidth * 0.03;
        const centerY = badgeY + badgeHeight / 2;
        const gap = badgeWidth * 0.15;
        const totalWidth = logoWidth + gap + textSize * 3.6;

        const labelGroup = nodeG
          .append("g")
          .attr("transform", `rotate(90 ${centerX} ${centerY})`);

        labelGroup
          .append("svg")
          .attr("x", centerX - totalWidth / 2)
          .attr("y", centerY - logoHeight / 2)
          .attr("width", logoWidth)
          .attr("height", logoHeight)
          .attr("viewBox", "0 0 2.174 1.438")
          .append("path")
          .attr("d", NVIDIA_LOGO_PATH)
          .attr("fill", "#76b900");

        labelGroup
          .append("text")
          .attr("x", centerX - totalWidth / 2 + logoWidth + gap)
          .attr("y", centerY)
          .attr("text-anchor", "start")
          .attr("dominant-baseline", "middle")
          .attr("fill", "#8a7a56")
          .attr("font-size", textSize)
          .attr("font-family", "monospace")
          .attr("font-weight", "700")
          .text("NVIDIA");
      } else if (isLinux) {
        // Linux — Tux penguin
        iconBaseWidth = nodeRadius * 1.0;
        iconBaseHeight = nodeRadius * 1.1;
        const tuxSize = Math.min(iconBaseWidth, iconBaseHeight);
        const scale = tuxSize / 100;
        const tx = nodeInfo.x - (100 * scale) / 2;
        const ty = nodeInfo.y - (100 * scale) / 2;

        // Body (dark outline)
        nodeG
          .append("path")
          .attr("class", "node-outline")
          .attr("d", TUX_BODY_PATH)
          .attr("transform", `translate(${tx}, ${ty}) scale(${scale})`)
          .attr("fill", "#1a1a1a")
          .attr("stroke", wireColor)
          .attr("stroke-width", strokeWidth / scale);

        // White belly
        nodeG
          .append("path")
          .attr("d", TUX_BELLY_PATH)
          .attr("transform", `translate(${tx}, ${ty}) scale(${scale})`)
          .attr("fill", "rgba(220,220,220,0.85)");

        // Memory fill on belly (from bottom up)
        if (ramUsagePercent > 0) {
          const bellyClipId = `tux-belly-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
          defs
            .append("clipPath")
            .attr("id", bellyClipId)
            .append("path")
            .attr("d", TUX_BELLY_PATH)
            .attr("transform", `translate(${tx}, ${ty}) scale(${scale})`);

          const bellyTop = ty + 37 * scale;
          const bellyHeight = 36 * scale;
          const memHeight = (ramUsagePercent / 100) * bellyHeight;
          nodeG
            .append("rect")
            .attr("x", tx + 38 * scale)
            .attr("y", bellyTop + bellyHeight - memHeight)
            .attr("width", 24 * scale)
            .attr("height", memHeight)
            .attr("fill", "rgba(255,215,0,0.75)")
            .attr("clip-path", `url(#${bellyClipId})`);
        }

        // Eyes
        const eyeRadius = 2.5 * scale;
        nodeG
          .append("circle")
          .attr("cx", tx + 44 * scale)
          .attr("cy", ty + 16 * scale)
          .attr("r", eyeRadius)
          .attr("fill", "white");
        nodeG
          .append("circle")
          .attr("cx", tx + 56 * scale)
          .attr("cy", ty + 16 * scale)
          .attr("r", eyeRadius)
          .attr("fill", "white");
        // Pupils
        nodeG
          .append("circle")
          .attr("cx", tx + 44 * scale)
          .attr("cy", ty + 16 * scale)
          .attr("r", eyeRadius * 0.5)
          .attr("fill", "#1a1a1a");
        nodeG
          .append("circle")
          .attr("cx", tx + 56 * scale)
          .attr("cy", ty + 16 * scale)
          .attr("r", eyeRadius * 0.5)
          .attr("fill", "#1a1a1a");

        // Beak
        nodeG
          .append("path")
          .attr(
            "d",
            `M${tx + 46 * scale} ${ty + 22 * scale} L${tx + 50 * scale} ${ty + 27 * scale} L${tx + 54 * scale} ${ty + 22 * scale} Z`,
          )
          .attr("fill", "#E8A317");

        // Feet
        nodeG
          .append("ellipse")
          .attr("cx", tx + 42 * scale)
          .attr("cy", ty + 94 * scale)
          .attr("rx", 6 * scale)
          .attr("ry", 2.5 * scale)
          .attr("fill", "#E8A317");
        nodeG
          .append("ellipse")
          .attr("cx", tx + 58 * scale)
          .attr("cy", ty + 94 * scale)
          .attr("rx", 6 * scale)
          .attr("ry", 2.5 * scale)
          .attr("fill", "#E8A317");
      } else if (modelLower === "mac studio") {
        // Mac Studio - classic cube with memory fill
        iconBaseWidth = nodeRadius * 1.25;
        iconBaseHeight = nodeRadius * 0.85;
        const x = nodeInfo.x - iconBaseWidth / 2;
        const y = nodeInfo.y - iconBaseHeight / 2;
        const cornerRadius = 4;
        const topSurfaceHeight = iconBaseHeight * 0.15;

        // Create clip path for memory fill area (front body)
        const studioClipId = `studio-clip-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
        defs
          .append("clipPath")
          .attr("id", studioClipId)
          .append("rect")
          .attr("x", x)
          .attr("y", y + topSurfaceHeight)
          .attr("width", iconBaseWidth)
          .attr("height", iconBaseHeight - topSurfaceHeight)
          .attr("rx", cornerRadius - 1);

        // Main body (uniform color)
        nodeG
          .append("rect")
          .attr("class", "node-outline")
          .attr("x", x)
          .attr("y", y)
          .attr("width", iconBaseWidth)
          .attr("height", iconBaseHeight)
          .attr("rx", cornerRadius)
          .attr("fill", "#1a1a1a")
          .attr("stroke", wireColor)
          .attr("stroke-width", strokeWidth);

        // Memory fill (fills from bottom up)
        if (ramUsagePercent > 0) {
          const memFillTotalHeight = iconBaseHeight - topSurfaceHeight;
          const memFillActualHeight =
            (ramUsagePercent / 100) * memFillTotalHeight;
          nodeG
            .append("rect")
            .attr("x", x)
            .attr(
              "y",
              y + topSurfaceHeight + (memFillTotalHeight - memFillActualHeight),
            )
            .attr("width", iconBaseWidth)
            .attr("height", memFillActualHeight)
            .attr("fill", "rgba(255,215,0,0.75)")
            .attr("clip-path", `url(#${studioClipId})`);
        }

        // Front panel details - vertical slots
        const detailColor = "rgba(0,0,0,0.35)";
        const slotHeight = iconBaseHeight * 0.14;
        const vSlotWidth = iconBaseWidth * 0.05;
        const vSlotY =
          y + topSurfaceHeight + (iconBaseHeight - topSurfaceHeight) * 0.6;
        const vSlot1X = x + iconBaseWidth * 0.18;
        const vSlot2X = x + iconBaseWidth * 0.28;

        [vSlot1X, vSlot2X].forEach((vx) => {
          nodeG
            .append("rect")
            .attr("x", vx - vSlotWidth / 2)
            .attr("y", vSlotY)
            .attr("width", vSlotWidth)
            .attr("height", slotHeight)
            .attr("fill", detailColor)
            .attr("rx", 1.5);
        });

        // Horizontal slot (SD card)
        const hSlotWidth = iconBaseWidth * 0.2;
        const hSlotX = x + iconBaseWidth * 0.5 - hSlotWidth / 2;
        nodeG
          .append("rect")
          .attr("x", hSlotX)
          .attr("y", vSlotY)
          .attr("width", hSlotWidth)
          .attr("height", slotHeight * 0.6)
          .attr("fill", detailColor)
          .attr("rx", 1);
      } else if (modelLower === "mac mini") {
        // Mac Mini - classic flat box with memory fill
        iconBaseWidth = nodeRadius * 1.3;
        iconBaseHeight = nodeRadius * 0.7;
        const x = nodeInfo.x - iconBaseWidth / 2;
        const y = nodeInfo.y - iconBaseHeight / 2;
        const cornerRadius = 3;
        const topSurfaceHeight = iconBaseHeight * 0.2;

        // Create clip path for memory fill area
        const miniClipId = `mini-clip-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
        defs
          .append("clipPath")
          .attr("id", miniClipId)
          .append("rect")
          .attr("x", x)
          .attr("y", y + topSurfaceHeight)
          .attr("width", iconBaseWidth)
          .attr("height", iconBaseHeight - topSurfaceHeight)
          .attr("rx", cornerRadius - 1);

        // Main body (uniform color)
        nodeG
          .append("rect")
          .attr("class", "node-outline")
          .attr("x", x)
          .attr("y", y)
          .attr("width", iconBaseWidth)
          .attr("height", iconBaseHeight)
          .attr("rx", cornerRadius)
          .attr("fill", "#1a1a1a")
          .attr("stroke", wireColor)
          .attr("stroke-width", strokeWidth);

        // Memory fill (fills from bottom up)
        if (ramUsagePercent > 0) {
          const memFillTotalHeight = iconBaseHeight - topSurfaceHeight;
          const memFillActualHeight =
            (ramUsagePercent / 100) * memFillTotalHeight;
          nodeG
            .append("rect")
            .attr("x", x)
            .attr(
              "y",
              y + topSurfaceHeight + (memFillTotalHeight - memFillActualHeight),
            )
            .attr("width", iconBaseWidth)
            .attr("height", memFillActualHeight)
            .attr("fill", "rgba(255,215,0,0.75)")
            .attr("clip-path", `url(#${miniClipId})`);
        }

        // Front panel details - vertical slots (no horizontal slot for Mini)
        const detailColor = "rgba(0,0,0,0.35)";
        const slotHeight = iconBaseHeight * 0.2;
        const vSlotWidth = iconBaseWidth * 0.045;
        const vSlotY =
          y + topSurfaceHeight + (iconBaseHeight - topSurfaceHeight) * 0.45;
        const vSlot1X = x + iconBaseWidth * 0.2;
        const vSlot2X = x + iconBaseWidth * 0.3;

        [vSlot1X, vSlot2X].forEach((vx) => {
          nodeG
            .append("rect")
            .attr("x", vx - vSlotWidth / 2)
            .attr("y", vSlotY)
            .attr("width", vSlotWidth)
            .attr("height", slotHeight)
            .attr("fill", detailColor)
            .attr("rx", 1.2);
        });
      } else if (
        modelLower === "macbook pro" ||
        modelLower.includes("macbook")
      ) {
        // MacBook Pro - classic style with memory fill on screen
        iconBaseWidth = nodeRadius * 1.6;
        iconBaseHeight = nodeRadius * 1.15;
        const x = nodeInfo.x - iconBaseWidth / 2;
        const y = nodeInfo.y - iconBaseHeight / 2;

        const screenHeight = iconBaseHeight * 0.7;
        const baseHeight = iconBaseHeight * 0.3;
        const screenWidth = iconBaseWidth * 0.85;
        const screenX = nodeInfo.x - screenWidth / 2;
        const screenBezel = 3;

        // Create clip path for screen content
        const screenClipId = `screen-clip-${nodeInfo.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
        defs
          .append("clipPath")
          .attr("id", screenClipId)
          .append("rect")
          .attr("x", screenX + screenBezel)
          .attr("y", y + screenBezel)
          .attr("width", screenWidth - screenBezel * 2)
          .attr("height", screenHeight - screenBezel * 2)
          .attr("rx", 2);

        // Screen outer frame
        nodeG
          .append("rect")
          .attr("class", "node-outline")
          .attr("x", screenX)
          .attr("y", y)
          .attr("width", screenWidth)
          .attr("height", screenHeight)
          .attr("rx", 3)
          .attr("fill", "#1a1a1a")
          .attr("stroke", wireColor)
          .attr("stroke-width", strokeWidth);

        // Screen inner (dark background)
        nodeG
          .append("rect")
          .attr("x", screenX + screenBezel)
          .attr("y", y + screenBezel)
          .attr("width", screenWidth - screenBezel * 2)
          .attr("height", screenHeight - screenBezel * 2)
          .attr("rx", 2)
          .attr("fill", "#0a0a12");

        // Memory fill on screen (fills from bottom up - classic style)
        if (ramUsagePercent > 0) {
          const memFillTotalHeight = screenHeight - screenBezel * 2;
          const memFillActualHeight =
            (ramUsagePercent / 100) * memFillTotalHeight;
          nodeG
            .append("rect")
            .attr("x", screenX + screenBezel)
            .attr(
              "y",
              y + screenBezel + (memFillTotalHeight - memFillActualHeight),
            )
            .attr("width", screenWidth - screenBezel * 2)
            .attr("height", memFillActualHeight)
            .attr("fill", "rgba(255,215,0,0.85)")
            .attr("clip-path", `url(#${screenClipId})`);
        }

        // Apple logo on screen (centered, on top of memory fill)
        const targetLogoHeight = screenHeight * 0.22;
        const logoScale = targetLogoHeight / LOGO_NATIVE_HEIGHT;
        const logoX = nodeInfo.x - (LOGO_NATIVE_WIDTH * logoScale) / 2;
        const logoY =
          y + screenHeight / 2 - (LOGO_NATIVE_HEIGHT * logoScale) / 2;
        nodeG
          .append("path")
          .attr("d", APPLE_LOGO_PATH)
          .attr(
            "transform",
            `translate(${logoX}, ${logoY}) scale(${logoScale})`,
          )
          .attr("fill", "#FFFFFF")
          .attr("opacity", 0.9);

        // Base (keyboard) - trapezoidal
        const baseY = y + screenHeight;
        const baseTopWidth = screenWidth;
        const baseBottomWidth = iconBaseWidth;
        const baseTopX = nodeInfo.x - baseTopWidth / 2;
        const baseBottomX = nodeInfo.x - baseBottomWidth / 2;

        nodeG
          .append("path")
          .attr(
            "d",
            `M ${baseTopX} ${baseY} L ${baseTopX + baseTopWidth} ${baseY} L ${baseBottomX + baseBottomWidth} ${baseY + baseHeight} L ${baseBottomX} ${baseY + baseHeight} Z`,
          )
          .attr("fill", "#2c2c2c")
          .attr("stroke", wireColor)
          .attr("stroke-width", 1);

        // Keyboard area
        const keyboardX = baseTopX + 6;
        const keyboardY = baseY + 3;
        const keyboardWidth = baseTopWidth - 12;
        const keyboardHeight = baseHeight * 0.55;
        nodeG
          .append("rect")
          .attr("x", keyboardX)
          .attr("y", keyboardY)
          .attr("width", keyboardWidth)
          .attr("height", keyboardHeight)
          .attr("fill", "rgba(0,0,0,0.2)")
          .attr("rx", 2);

        // Trackpad
        const trackpadWidth = baseTopWidth * 0.4;
        const trackpadX = nodeInfo.x - trackpadWidth / 2;
        const trackpadY = baseY + keyboardHeight + 5;
        const trackpadHeight = baseHeight * 0.3;
        nodeG
          .append("rect")
          .attr("x", trackpadX)
          .attr("y", trackpadY)
          .attr("width", trackpadWidth)
          .attr("height", trackpadHeight)
          .attr("fill", "rgba(255,255,255,0.08)")
          .attr("rx", 2);
      } else {
        // Default/Unknown - holographic hexagon
        const hexRadius = nodeRadius * 0.6;
        const hexPoints = Array.from({ length: 6 }, (_, i) => {
          const angle = ((i * 60 - 30) * Math.PI) / 180;
          return `${nodeInfo.x + hexRadius * Math.cos(angle)},${nodeInfo.y + hexRadius * Math.sin(angle)}`;
        }).join(" ");

        // Main shape
        nodeG
          .append("polygon")
          .attr("class", "node-outline")
          .attr("points", hexPoints)
          .attr("fill", fillColor)
          .attr("stroke", wireColor)
          .attr("stroke-width", strokeWidth);
      }

      // --- Vertical GPU Bar (right side of icon) ---
      // Show in both full mode and minimized mode (scaled appropriately)
      if (showFullLabels || isMinimized) {
        const gpuBarWidth = isMinimized
          ? Math.max(16, nodeRadius * 0.32)
          : Math.max(28, nodeRadius * 0.3);
        const gpuBarHeight = iconBaseHeight * 0.95;
        const barXOffset = iconBaseWidth / 2 + (isMinimized ? 5 : 10);
        const gpuBarX = nodeInfo.x + barXOffset;
        const gpuBarY = nodeInfo.y - gpuBarHeight / 2;

        // GPU Bar Background (grey, no border)
        nodeG
          .append("rect")
          .attr("x", gpuBarX)
          .attr("y", gpuBarY)
          .attr("width", gpuBarWidth)
          .attr("height", gpuBarHeight)
          .attr("fill", "rgba(80, 80, 90, 0.7)")
          .attr("rx", 2);

        // GPU Bar Fill (from bottom up, colored by temperature)
        if (gpuUsagePercent > 0) {
          const fillHeight = (gpuUsagePercent / 100) * gpuBarHeight;
          const gpuFillColor = getTemperatureColor(gpuTemp);
          nodeG
            .append("rect")
            .attr("x", gpuBarX)
            .attr("y", gpuBarY + (gpuBarHeight - fillHeight))
            .attr("width", gpuBarWidth)
            .attr("height", fillHeight)
            .attr("fill", gpuFillColor)
            .attr("opacity", 0.9)
            .attr("rx", 2);
        }

        // GPU Stats Text (centered on bar, multiline, bigger and bold)
        const gpuTextX = gpuBarX + gpuBarWidth / 2;
        const gpuTextY = gpuBarY + gpuBarHeight / 2;
        const gpuTextFontSize = isMinimized
          ? Math.max(10, gpuBarWidth * 0.6)
          : Math.min(16, Math.max(12, gpuBarWidth * 0.55));
        const lineSpacing = gpuTextFontSize * 1.25;

        const gpuUsageText = `${gpuUsagePercent.toFixed(0)}%`;
        const tempText = !isNaN(gpuTemp) ? `${gpuTemp.toFixed(0)}°C` : "-";
        const powerText = sysPower !== null ? `${sysPower.toFixed(0)}W` : "-";

        // GPU Usage %
        nodeG
          .append("text")
          .attr("x", gpuTextX)
          .attr("y", gpuTextY - lineSpacing)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", "#FFFFFF")
          .attr("font-size", gpuTextFontSize)
          .attr("font-weight", "700")
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(gpuUsageText);

        // Temperature
        nodeG
          .append("text")
          .attr("x", gpuTextX)
          .attr("y", gpuTextY)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", "#FFFFFF")
          .attr("font-size", gpuTextFontSize)
          .attr("font-weight", "700")
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(tempText);

        // Power (Watts)
        nodeG
          .append("text")
          .attr("x", gpuTextX)
          .attr("y", gpuTextY + lineSpacing)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", "#FFFFFF")
          .attr("font-size", gpuTextFontSize)
          .attr("font-weight", "700")
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(powerText);
      }

      // Labels - adapt based on mode
      if (showFullLabels) {
        // FULL MODE: Name above, memory info below (1-4 nodes)
        const nameY = nodeInfo.y - iconBaseHeight / 2 - 15;
        const fontSize = Math.max(10, nodeRadius * 0.16);

        // Truncate name based on node count
        const maxNameLen =
          numNodes === 1 ? 22 : numNodes === 2 ? 18 : numNodes === 3 ? 16 : 14;
        const displayName =
          friendlyName.length > maxNameLen
            ? friendlyName.slice(0, maxNameLen - 2) + ".."
            : friendlyName;

        // Name label above
        nodeG
          .append("text")
          .attr("x", nodeInfo.x)
          .attr("y", nameY)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", "#FFD700")
          .attr("font-size", fontSize)
          .attr("font-weight", 500)
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(displayName);

        // Memory info below - used in grey, total in yellow
        const infoY = nodeInfo.y + iconBaseHeight / 2 + 16;
        const memText = nodeG
          .append("text")
          .attr("x", nodeInfo.x)
          .attr("y", infoY)
          .attr("text-anchor", "middle")
          .attr("font-size", fontSize * 0.85)
          .attr("font-family", "SF Mono, Monaco, monospace");
        memText
          .append("tspan")
          .attr("fill", "rgba(255,215,0,0.9)")
          .text(`${formatBytes(ramUsed)}`);
        memText
          .append("tspan")
          .attr("fill", "rgba(179,179,179,0.9)")
          .text(`/${formatBytes(ramTotal)}`);
        memText
          .append("tspan")
          .attr("fill", "rgba(179,179,179,0.7)")
          .text(` (${ramUsagePercent.toFixed(0)}%)`);
      } else if (showCompactLabels) {
        // COMPACT MODE: Just name and basic info (4+ nodes)
        const fontSize = Math.max(7, nodeRadius * 0.11);

        // Very compact name below icon
        const nameY = nodeInfo.y + iconBaseHeight / 2 + 9;
        const shortName =
          friendlyName.length > 10
            ? friendlyName.slice(0, 8) + ".."
            : friendlyName;
        nodeG
          .append("text")
          .attr("x", nodeInfo.x)
          .attr("y", nameY)
          .attr("text-anchor", "middle")
          .attr("fill", "#FFD700")
          .attr("font-size", fontSize)
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(shortName);

        // Single line of key stats
        const statsY = nameY + 9;
        nodeG
          .append("text")
          .attr("x", nodeInfo.x)
          .attr("y", statsY)
          .attr("text-anchor", "middle")
          .attr("fill", "rgba(255,215,0,0.7)")
          .attr("font-size", fontSize * 0.85)
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(
            `${ramUsagePercent.toFixed(0)}%${!isNaN(gpuTemp) ? " " + gpuTemp.toFixed(0) + "°C" : ""}`,
          );
      } else {
        // MINIMIZED MODE: Show name above and memory info below (like main topology)
        const fontSize = 8;

        // Friendly name (shortened) above icon
        const nameY = nodeInfo.y - iconBaseHeight / 2 - 8;
        const shortName =
          friendlyName.length > 12
            ? friendlyName.slice(0, 10) + ".."
            : friendlyName;
        nodeG
          .append("text")
          .attr("x", nodeInfo.x)
          .attr("y", nameY)
          .attr("text-anchor", "middle")
          .attr("fill", "#FFD700")
          .attr("font-size", fontSize)
          .attr("font-weight", "500")
          .attr("font-family", "SF Mono, Monaco, monospace")
          .text(shortName);

        // Memory info below icon - used in grey, total in yellow (same as main topology)
        const infoY = nodeInfo.y + iconBaseHeight / 2 + 10;
        const memTextMini = nodeG
          .append("text")
          .attr("x", nodeInfo.x)
          .attr("y", infoY)
          .attr("text-anchor", "middle")
          .attr("font-size", fontSize * 0.85)
          .attr("font-family", "SF Mono, Monaco, monospace");
        memTextMini
          .append("tspan")
          .attr("fill", "rgba(255,215,0,0.9)")
          .text(`${formatBytes(ramUsed)}`);
        memTextMini
          .append("tspan")
          .attr("fill", "rgba(179,179,179,0.9)")
          .text(`/${formatBytes(ramTotal)}`);
        memTextMini
          .append("tspan")
          .attr("fill", "rgba(179,179,179,0.7)")
          .text(` (${ramUsagePercent.toFixed(0)}%)`);
      }

      // Debug mode: Show TB bridge and RDMA status
      if (debugEnabled) {
        let debugLabelY =
          nodeInfo.y +
          iconBaseHeight / 2 +
          (showFullLabels ? 32 : showCompactLabels ? 26 : 22);
        const debugFontSize = showFullLabels ? 9 : 7;
        const debugLineHeight = showFullLabels ? 11 : 9;

        const tbStatus = tbBridgeData[nodeInfo.id];
        if (tbStatus) {
          const tbColor = tbStatus.enabled
            ? "rgba(234,179,8,0.9)"
            : "rgba(100,100,100,0.7)";
          const tbText = tbStatus.enabled ? "TB:ON" : "TB:OFF";
          nodeG
            .append("text")
            .attr("x", nodeInfo.x)
            .attr("y", debugLabelY)
            .attr("text-anchor", "middle")
            .attr("fill", tbColor)
            .attr("font-size", debugFontSize)
            .attr("font-family", "SF Mono, Monaco, monospace")
            .text(tbText);
          debugLabelY += debugLineHeight;
        }

        const rdmaStatus = rdmaCtlData[nodeInfo.id];
        if (rdmaStatus !== undefined) {
          const rdmaColor = rdmaStatus.enabled
            ? "rgba(74,222,128,0.9)"
            : "rgba(100,100,100,0.7)";
          const rdmaText = rdmaStatus.enabled ? "RDMA:ON" : "RDMA:OFF";
          nodeG
            .append("text")
            .attr("x", nodeInfo.x)
            .attr("y", debugLabelY)
            .attr("text-anchor", "middle")
            .attr("fill", rdmaColor)
            .attr("font-size", debugFontSize)
            .attr("font-family", "SF Mono, Monaco, monospace")
            .text(rdmaText);
          debugLabelY += debugLineHeight;
        }

        const dbgIdentity = identitiesData[nodeInfo.id];
        if (dbgIdentity?.osVersion) {
          const osLabel =
            dbgIdentity.osVersion === "Linux"
              ? "Linux"
              : `macOS ${dbgIdentity.osVersion}${dbgIdentity.osBuildVersion ? ` (${dbgIdentity.osBuildVersion})` : ""}`;
          nodeG
            .append("text")
            .attr("x", nodeInfo.x)
            .attr("y", debugLabelY)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(179,179,179,0.7)")
            .attr("font-size", debugFontSize)
            .attr("font-family", "SF Mono, Monaco, monospace")
            .text(osLabel);
        }
      }
    });
  }

  $effect(() => {
    // Track all reactive dependencies that affect rendering
    const _data = data;
    const _hoveredNodeId = hoveredNodeId;
    const _filteredNodes = filteredNodes;
    const _highlightedNodes = highlightedNodes;
    if (_data) {
      renderGraph();
    }
  });

  onMount(() => {
    if (svgContainer) {
      resizeObserver = new ResizeObserver(() => {
        renderGraph();
      });
      resizeObserver.observe(svgContainer);
    }
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
  });
</script>

<svg bind:this={svgContainer} class="w-full h-full {className}"></svg>

<style>
  :global(.graph-node) {
    /* Only transition opacity for filtered-out nodes, no transition on hover stroke changes */
    transition: opacity 0.2s ease;
  }
  :global(.graph-link) {
    stroke: var(--exo-light-gray, #b3b3b3);
    stroke-width: 1px;
    stroke-dasharray: 4, 4;
    opacity: 0.8;
    animation: flowAnimation 0.75s linear infinite;
  }
  @keyframes flowAnimation {
    from {
      stroke-dashoffset: 0;
    }
    to {
      stroke-dashoffset: -10;
    }
  }
</style>
