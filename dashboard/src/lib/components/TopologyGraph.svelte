<script lang="ts">
	import { topologyData, isTopologyMinimized, debugMode } from '$lib/stores/app.svelte';

	interface Props {
		class?: string;
		highlightedNodes?: Set<string>;
	}

	let { class: className = '', highlightedNodes = new Set() }: Props = $props();

	// Track container dimensions reactively
	let containerWidth = $state(0);
	let containerHeight = $state(0);

	const isMinimized = $derived(isTopologyMinimized());
	const data = $derived(topologyData());
	const debugEnabled = $derived(debugMode());

	// Apple logo path for MacBook Pro screen
	const APPLE_LOGO_PATH = "M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z";
	const LOGO_NATIVE_WIDTH = 814;
	const LOGO_NATIVE_HEIGHT = 1000;

	function formatBytes(bytes: number, decimals = 1): string {
		if (!bytes || bytes === 0) return '0B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + sizes[i];
	}

	function getTemperatureColor(temp: number): string {
		if (isNaN(temp) || temp === null) return 'rgba(179, 179, 179, 0.8)';
		
		const coolTemp = 45;
		const midTemp = 57.5;
		const hotTemp = 75;
		
		const coolColor = { r: 93, g: 173, b: 226 };
		const midColor = { r: 255, g: 215, b: 0 };
		const hotColor = { r: 244, g: 67, b: 54 };
		
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

	function getInterfaceLabel(nodeId: string, ip?: string): { label: string; missing: boolean } {
		if (!ip) return { label: '?', missing: true };
		const cleanIp = ip.includes(':') && !ip.includes('[') ? ip.split(':')[0] : ip;
		
		function checkNode(node: typeof data.nodes[string]): string | null {
			if (!node) return null;
			const matchFromInterfaces = node.network_interfaces?.find((iface) =>
				(iface.addresses || []).some((addr) => addr === cleanIp || addr === ip)
			);
			if (matchFromInterfaces?.name) return matchFromInterfaces.name;
			const mapped = node.ip_to_interface?.[cleanIp] || node.ip_to_interface?.[ip];
			if (mapped && mapped.trim().length > 0) return mapped;
			return null;
		}
		
		const result = checkNode(data?.nodes?.[nodeId]);
		if (result) return { label: result, missing: false };
		
		for (const [, otherNode] of Object.entries(data?.nodes || {})) {
			const otherResult = checkNode(otherNode);
			if (otherResult) return { label: otherResult, missing: false };
		}
		return { label: '?', missing: true };
	}

	// Computed layout values
	const layout = $derived(() => {
		if (!data || containerWidth === 0 || containerHeight === 0) {
			return { nodes: [], edges: [], centerX: 0, centerY: 0, nodeRadius: 0, showFullLabels: false, showCompactLabels: false };
		}

		const nodes = data.nodes || {};
		const edges = data.edges || [];
		const nodeIds = Object.keys(nodes);
		const numNodes = nodeIds.length;

		const width = containerWidth;
		const height = containerHeight;
		const centerX = width / 2;
		const minDimension = Math.min(width, height);

		// Dynamic scaling
		const sizeScale = numNodes === 1 ? 1 : Math.max(0.6, 1 - (numNodes - 1) * 0.10);
		const baseNodeRadius = isMinimized
			? Math.max(36, Math.min(60, minDimension * 0.22))
			: Math.min(120, minDimension * 0.20);
		const nodeRadius = baseNodeRadius * sizeScale;

		// Orbit radius
		const circumference = numNodes * nodeRadius * 4;
		const radiusFromCircumference = circumference / (2 * Math.PI);
		const minOrbitRadius = Math.max(radiusFromCircumference, minDimension * 0.18);
		const maxOrbitRadius = minDimension * 0.30;
		const orbitRadius = isMinimized
			? Math.min(maxOrbitRadius, Math.max(minOrbitRadius, minDimension * 0.26))
			: Math.min(maxOrbitRadius, Math.max(minOrbitRadius, minDimension * (0.22 + numNodes * 0.02)));

		const showFullLabels = !isMinimized && numNodes <= 4;
		const showCompactLabels = !isMinimized && numNodes > 4;

		const topPadding = 70;
		const bottomPadding = 70;
		const safeCenterY = topPadding + (height - topPadding - bottomPadding) / 2;

		// Calculate node positions and metrics
		const nodesWithPositions = nodeIds.map((id, index) => {
			const nodeData = nodes[id];
			const macmon = nodeData.macmon_info;
			const modelId = nodeData.system_info?.model_id || 'Unknown';
			const friendlyName = nodeData.friendly_name || modelId;

			let ramUsagePercent = 0;
			let gpuTemp = NaN;
			let ramTotal = 0;
			let ramUsed = 0;
			let gpuUsagePercent = 0;
			let sysPower: number | null = null;

			if (macmon) {
				if (macmon.memory && macmon.memory.ram_total > 0) {
					ramUsagePercent = (macmon.memory.ram_usage / macmon.memory.ram_total) * 100;
					ramTotal = macmon.memory.ram_total;
					ramUsed = macmon.memory.ram_usage;
				}
				if (macmon.temp && typeof macmon.temp.gpu_temp_avg === 'number') {
					gpuTemp = Math.max(30, macmon.temp.gpu_temp_avg);
				}
				if (macmon.gpu_usage) {
					gpuUsagePercent = macmon.gpu_usage[1] * 100;
				}
				if (macmon.sys_power) {
					sysPower = macmon.sys_power;
				}
			}

			let x: number, y: number;
			if (numNodes === 1) {
				x = centerX;
				y = safeCenterY;
			} else {
				const angle = (index / numNodes) * 2 * Math.PI - (Math.PI / 2);
				x = centerX + orbitRadius * Math.cos(angle);
				y = safeCenterY + orbitRadius * Math.sin(angle);
			}

			const isHighlighted = highlightedNodes.has(id);
			const modelLower = modelId.toLowerCase();
			const deviceType = modelLower === 'mac studio' ? 'studio' 
				: modelLower === 'mac mini' ? 'mini'
				: modelLower === 'macbook pro' || modelLower.includes('macbook') ? 'macbook'
				: 'unknown';

			// Icon dimensions based on device type
			let iconWidth = nodeRadius * 1.2;
			let iconHeight = nodeRadius * 1.0;
			if (deviceType === 'studio') {
				iconWidth = nodeRadius * 1.25;
				iconHeight = nodeRadius * 0.85;
			} else if (deviceType === 'mini') {
				iconWidth = nodeRadius * 1.3;
				iconHeight = nodeRadius * 0.7;
			} else if (deviceType === 'macbook') {
				iconWidth = nodeRadius * 1.6;
				iconHeight = nodeRadius * 1.15;
			}

			const wireColor = isHighlighted ? 'rgba(255,215,0,0.9)' : 'rgba(179,179,179,0.8)';
			const strokeWidth = isHighlighted ? 2.5 : 1.5;

			// Truncate name based on mode
			let displayName = friendlyName;
			if (showFullLabels) {
				const maxLen = numNodes === 1 ? 22 : (numNodes === 2 ? 18 : numNodes === 3 ? 16 : 14);
				displayName = friendlyName.length > maxLen ? friendlyName.slice(0, maxLen - 2) + '..' : friendlyName;
			} else if (showCompactLabels) {
				displayName = friendlyName.length > 10 ? friendlyName.slice(0, 8) + '..' : friendlyName;
			} else {
				displayName = friendlyName.length > 12 ? friendlyName.slice(0, 10) + '..' : friendlyName;
			}

			return {
				id,
				x,
				y,
				deviceType,
				iconWidth,
				iconHeight,
				modelId,
				friendlyName,
				displayName,
				ramUsagePercent,
				ramUsed,
				ramTotal,
				gpuUsagePercent,
				gpuTemp,
				sysPower,
				isHighlighted,
				wireColor,
				strokeWidth,
				gpuFillColor: getTemperatureColor(gpuTemp),
				ramUsedFormatted: formatBytes(ramUsed),
				ramTotalFormatted: formatBytes(ramTotal)
			};
		});

		// Build position lookup for edges
		const positionById: Record<string, { x: number; y: number }> = {};
		nodesWithPositions.forEach(n => { positionById[n.id] = { x: n.x, y: n.y }; });

		// Process edges into pairs with direction info
		const pairMap = new Map<string, { a: string; b: string; aToB: boolean; bToA: boolean; connections: Array<{ from: string; to: string; ip: string; ifaceLabel: string; missingIface: boolean }> }>();
		
		edges.forEach(edge => {
			if (!edge.source || !edge.target || edge.source === edge.target) return;
			if (!positionById[edge.source] || !positionById[edge.target]) return;
			
			const a = edge.source < edge.target ? edge.source : edge.target;
			const b = edge.source < edge.target ? edge.target : edge.source;
			const key = `${a}|${b}`;
			const entry = pairMap.get(key) || { a, b, aToB: false, bToA: false, connections: [] };
			
			if (edge.source === a) entry.aToB = true;
			else entry.bToA = true;

			const ip = edge.sendBackIp || '?';
			const ifaceInfo = getInterfaceLabel(edge.source, ip);
			entry.connections.push({
				from: edge.source,
				to: edge.target,
				ip,
				ifaceLabel: ifaceInfo.label,
				missingIface: ifaceInfo.missing
			});
			pairMap.set(key, entry);
		});

		// Convert edge pairs to renderable format
		const edgeData = Array.from(pairMap.values()).map(entry => {
			const posA = positionById[entry.a];
			const posB = positionById[entry.b];
			if (!posA || !posB) return null;

			const dx = posB.x - posA.x;
			const dy = posB.y - posA.y;
			const len = Math.hypot(dx, dy) || 1;
			const ux = dx / len;
			const uy = dy / len;
			const mx = (posA.x + posB.x) / 2;
			const my = (posA.y + posB.y) / 2;
			const tipOffset = 16;
			const carrier = 2;

			return {
				key: `${entry.a}|${entry.b}`,
				x1: posA.x,
				y1: posA.y,
				x2: posB.x,
				y2: posB.y,
				aToB: entry.aToB,
				bToA: entry.bToA,
				arrowAtoB: entry.aToB ? {
					x1: mx - ux * tipOffset - ux * carrier,
					y1: my - uy * tipOffset - uy * carrier,
					x2: mx - ux * tipOffset,
					y2: my - uy * tipOffset
				} : null,
				arrowBtoA: entry.bToA ? {
					x1: mx + ux * tipOffset + ux * carrier,
					y1: my + uy * tipOffset + uy * carrier,
					x2: mx + ux * tipOffset,
					y2: my + uy * tipOffset
				} : null,
				connections: entry.connections,
				mx,
				my,
				isLeft: mx < centerX,
				isTop: my < safeCenterY
			};
		}).filter((e): e is NonNullable<typeof e> => e !== null);

		// Group debug labels by quadrant
		const debugLabels = debugEnabled ? {
			topLeft: edgeData.filter(e => e.isTop && e.isLeft).flatMap(e => e.connections),
			topRight: edgeData.filter(e => e.isTop && !e.isLeft).flatMap(e => e.connections),
			bottomLeft: edgeData.filter(e => !e.isTop && e.isLeft).flatMap(e => e.connections),
			bottomRight: edgeData.filter(e => !e.isTop && !e.isLeft).flatMap(e => e.connections)
		} : null;

		return {
			nodes: nodesWithPositions,
			edges: edgeData,
			centerX,
			centerY: safeCenterY,
			nodeRadius,
			showFullLabels,
			showCompactLabels,
			debugLabels,
			width,
			height,
			numNodes
		};
	});

	// Helper to get directional arrow character
	function getArrow(from: string, to: string): string {
		const nodes = layout().nodes;
		const fromNode = nodes.find(n => n.id === from);
		const toNode = nodes.find(n => n.id === to);
		if (!fromNode || !toNode) return '→';
		
		const dx = toNode.x - fromNode.x;
		const dy = toNode.y - fromNode.y;
		const absX = Math.abs(dx);
		const absY = Math.abs(dy);
		
		if (absX > absY * 2) return dx > 0 ? '→' : '←';
		if (absY > absX * 2) return dy > 0 ? '↓' : '↑';
		if (dx > 0 && dy > 0) return '↘';
		if (dx > 0 && dy < 0) return '↗';
		if (dx < 0 && dy > 0) return '↙';
		return '↖';
	}

	// GPU bar dimensions
	function getGpuBarDimensions(nodeRadius: number, iconWidth: number, iconHeight: number) {
		const gpuBarWidth = isMinimized ? Math.max(16, nodeRadius * 0.32) : Math.max(28, nodeRadius * 0.30);
		const gpuBarHeight = iconHeight * 0.95;
		const barXOffset = iconWidth / 2 + (isMinimized ? 5 : 10);
		return { gpuBarWidth, gpuBarHeight, barXOffset };
	}

	// Font size helpers
	function getLabelFontSize(nodeRadius: number, numNodes: number): number {
		const l = layout();
		if (l.showFullLabels) return Math.max(10, nodeRadius * 0.16);
		if (l.showCompactLabels) return Math.max(7, nodeRadius * 0.11);
		return 8;
	}
</script>

<svg 
	bind:clientWidth={containerWidth}
	bind:clientHeight={containerHeight}
	class="w-full h-full {className}"
>
	<!-- Defs for filters and markers -->
	<defs>
		<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
			<feGaussianBlur stdDeviation="2" result="coloredBlur"/>
			<feMerge>
				<feMergeNode in="coloredBlur"/>
				<feMergeNode in="SourceGraphic"/>
			</feMerge>
		</filter>
		
		<marker id="arrowhead" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="11" markerHeight="11" orient="auto-start-reverse">
			<path d="M 0 0 L 10 5 L 0 10" fill="none" stroke="var(--exo-light-gray, #B3B3B3)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
		</marker>
	</defs>

	{#if layout().nodes.length === 0}
		<!-- Empty state -->
		<text 
			x={containerWidth / 2} 
			y={containerHeight / 2} 
			text-anchor="middle" 
			dominant-baseline="middle"
			fill="rgba(255,215,0,0.4)"
			font-size={isMinimized ? 10 : 12}
			font-family="SF Mono, monospace"
			letter-spacing="0.1em"
		>AWAITING NODES</text>
	{:else}
		<!-- Edges -->
		{#each layout().edges as edge (edge.key)}
			<!-- Base dashed line -->
			<line 
				x1={edge.x1} y1={edge.y1} x2={edge.x2} y2={edge.y2}
				class="graph-link"
			/>
			
			<!-- Arrow A -> B -->
			{#if edge.arrowAtoB}
				<line 
					x1={edge.arrowAtoB.x1} y1={edge.arrowAtoB.y1} 
					x2={edge.arrowAtoB.x2} y2={edge.arrowAtoB.y2}
					stroke="none" fill="none"
					marker-end="url(#arrowhead)"
				/>
			{/if}
			
			<!-- Arrow B -> A -->
			{#if edge.arrowBtoA}
				<line 
					x1={edge.arrowBtoA.x1} y1={edge.arrowBtoA.y1} 
					x2={edge.arrowBtoA.x2} y2={edge.arrowBtoA.y2}
					stroke="none" fill="none"
					marker-end="url(#arrowhead)"
				/>
			{/if}
		{/each}

		<!-- Debug labels -->
		{#if layout().debugLabels}
			{@const padding = 10}
			{@const fontSize = isMinimized ? 10 : 12}
			{@const lineHeight = fontSize + 4}
			
			<!-- Top Left -->
			{#each layout().debugLabels.topLeft as conn, i}
				<text 
					x={padding} 
					y={padding + i * lineHeight}
					text-anchor="start"
					dominant-baseline="hanging"
					font-size={fontSize}
					font-family="SF Mono, monospace"
					fill={conn.missingIface ? 'rgba(248,113,113,0.9)' : 'rgba(255,255,255,0.85)'}
				>{getArrow(conn.from, conn.to)} {conn.ip} {conn.ifaceLabel}</text>
			{/each}
			
			<!-- Top Right -->
			{#each layout().debugLabels.topRight as conn, i}
				<text 
					x={layout().width - padding} 
					y={padding + i * lineHeight}
					text-anchor="end"
					dominant-baseline="hanging"
					font-size={fontSize}
					font-family="SF Mono, monospace"
					fill={conn.missingIface ? 'rgba(248,113,113,0.9)' : 'rgba(255,255,255,0.85)'}
				>{getArrow(conn.from, conn.to)} {conn.ip} {conn.ifaceLabel}</text>
			{/each}
			
			<!-- Bottom Left -->
			{#each layout().debugLabels.bottomLeft as conn, i}
				<text 
					x={padding} 
					y={layout().height - padding - (layout().debugLabels.bottomLeft.length - 1 - i) * lineHeight}
					text-anchor="start"
					font-size={fontSize}
					font-family="SF Mono, monospace"
					fill={conn.missingIface ? 'rgba(248,113,113,0.9)' : 'rgba(255,255,255,0.85)'}
				>{getArrow(conn.from, conn.to)} {conn.ip} {conn.ifaceLabel}</text>
			{/each}
			
			<!-- Bottom Right -->
			{#each layout().debugLabels.bottomRight as conn, i}
				<text 
					x={layout().width - padding} 
					y={layout().height - padding - (layout().debugLabels.bottomRight.length - 1 - i) * lineHeight}
					text-anchor="end"
					font-size={fontSize}
					font-family="SF Mono, monospace"
					fill={conn.missingIface ? 'rgba(248,113,113,0.9)' : 'rgba(255,255,255,0.85)'}
				>{getArrow(conn.from, conn.to)} {conn.ip} {conn.ifaceLabel}</text>
			{/each}
		{/if}

		<!-- Nodes -->
		{#each layout().nodes as node (node.id)}
			<g class="graph-node" style="cursor: pointer;">
				<title>{node.friendlyName}
ID: {node.id.slice(-8)}
Memory: {node.ramUsedFormatted}/{node.ramTotalFormatted}</title>

				{#if node.deviceType === 'studio'}
					<!-- Mac Studio -->
					{@const x = node.x - node.iconWidth / 2}
					{@const y = node.y - node.iconHeight / 2}
					{@const topSurfaceHeight = node.iconHeight * 0.15}
					{@const memFillTotalHeight = node.iconHeight - topSurfaceHeight}
					{@const memFillActualHeight = (node.ramUsagePercent / 100) * memFillTotalHeight}
					
					<clipPath id="studio-clip-{node.id}">
						<rect x={x} y={y + topSurfaceHeight} width={node.iconWidth} height={node.iconHeight - topSurfaceHeight} rx="3"/>
					</clipPath>
					
					<rect x={x} y={y} width={node.iconWidth} height={node.iconHeight} rx="4" fill="#1a1a1a" stroke={node.wireColor} stroke-width={node.strokeWidth}/>
					
					{#if node.ramUsagePercent > 0}
						<rect 
							x={x} 
							y={y + topSurfaceHeight + (memFillTotalHeight - memFillActualHeight)}
							width={node.iconWidth} 
							height={memFillActualHeight}
							fill="rgba(255,215,0,0.75)"
							clip-path="url(#studio-clip-{node.id})"
						/>
					{/if}
					
					<!-- Front panel slots -->
					{@const slotHeight = node.iconHeight * 0.14}
					{@const vSlotWidth = node.iconWidth * 0.05}
					{@const vSlotY = y + topSurfaceHeight + (node.iconHeight - topSurfaceHeight) * 0.6}
					<rect x={x + node.iconWidth * 0.18 - vSlotWidth / 2} y={vSlotY} width={vSlotWidth} height={slotHeight} fill="rgba(0,0,0,0.35)" rx="1.5"/>
					<rect x={x + node.iconWidth * 0.28 - vSlotWidth / 2} y={vSlotY} width={vSlotWidth} height={slotHeight} fill="rgba(0,0,0,0.35)" rx="1.5"/>
					<rect x={x + node.iconWidth * 0.5 - node.iconWidth * 0.1} y={vSlotY} width={node.iconWidth * 0.2} height={slotHeight * 0.6} fill="rgba(0,0,0,0.35)" rx="1"/>

				{:else if node.deviceType === 'mini'}
					<!-- Mac Mini -->
					{@const x = node.x - node.iconWidth / 2}
					{@const y = node.y - node.iconHeight / 2}
					{@const topSurfaceHeight = node.iconHeight * 0.20}
					{@const memFillTotalHeight = node.iconHeight - topSurfaceHeight}
					{@const memFillActualHeight = (node.ramUsagePercent / 100) * memFillTotalHeight}
					
					<clipPath id="mini-clip-{node.id}">
						<rect x={x} y={y + topSurfaceHeight} width={node.iconWidth} height={node.iconHeight - topSurfaceHeight} rx="2"/>
					</clipPath>
					
					<rect x={x} y={y} width={node.iconWidth} height={node.iconHeight} rx="3" fill="#1a1a1a" stroke={node.wireColor} stroke-width={node.strokeWidth}/>
					
					{#if node.ramUsagePercent > 0}
						<rect 
							x={x} 
							y={y + topSurfaceHeight + (memFillTotalHeight - memFillActualHeight)}
							width={node.iconWidth} 
							height={memFillActualHeight}
							fill="rgba(255,215,0,0.75)"
							clip-path="url(#mini-clip-{node.id})"
						/>
					{/if}
					
					<!-- Front panel slots -->
					{@const slotHeight = node.iconHeight * 0.20}
					{@const vSlotWidth = node.iconWidth * 0.045}
					{@const vSlotY = y + topSurfaceHeight + (node.iconHeight - topSurfaceHeight) * 0.45}
					<rect x={x + node.iconWidth * 0.20 - vSlotWidth / 2} y={vSlotY} width={vSlotWidth} height={slotHeight} fill="rgba(0,0,0,0.35)" rx="1.2"/>
					<rect x={x + node.iconWidth * 0.30 - vSlotWidth / 2} y={vSlotY} width={vSlotWidth} height={slotHeight} fill="rgba(0,0,0,0.35)" rx="1.2"/>

				{:else if node.deviceType === 'macbook'}
					<!-- MacBook Pro -->
					{@const x = node.x - node.iconWidth / 2}
					{@const y = node.y - node.iconHeight / 2}
					{@const screenHeight = node.iconHeight * 0.70}
					{@const baseHeight = node.iconHeight * 0.30}
					{@const screenWidth = node.iconWidth * 0.85}
					{@const screenX = node.x - screenWidth / 2}
					{@const screenBezel = 3}
					{@const memFillTotalHeight = screenHeight - screenBezel * 2}
					{@const memFillActualHeight = (node.ramUsagePercent / 100) * memFillTotalHeight}
					{@const logoScale = (screenHeight * 0.22) / LOGO_NATIVE_HEIGHT}
					{@const logoX = node.x - (LOGO_NATIVE_WIDTH * logoScale / 2)}
					{@const logoY = y + screenHeight / 2 - (LOGO_NATIVE_HEIGHT * logoScale / 2)}
					
					<clipPath id="screen-clip-{node.id}">
						<rect x={screenX + screenBezel} y={y + screenBezel} width={screenWidth - screenBezel * 2} height={screenHeight - screenBezel * 2} rx="2"/>
					</clipPath>
					
					<!-- Screen frame -->
					<rect x={screenX} y={y} width={screenWidth} height={screenHeight} rx="3" fill="#1a1a1a" stroke={node.wireColor} stroke-width={node.strokeWidth}/>
					<!-- Screen inner -->
					<rect x={screenX + screenBezel} y={y + screenBezel} width={screenWidth - screenBezel * 2} height={screenHeight - screenBezel * 2} rx="2" fill="#0a0a12"/>
					
					<!-- Memory fill -->
					{#if node.ramUsagePercent > 0}
						<rect 
							x={screenX + screenBezel}
							y={y + screenBezel + (memFillTotalHeight - memFillActualHeight)}
							width={screenWidth - screenBezel * 2}
							height={memFillActualHeight}
							fill="rgba(255,215,0,0.85)"
							clip-path="url(#screen-clip-{node.id})"
						/>
					{/if}
					
					<!-- Apple logo -->
					<path d={APPLE_LOGO_PATH} transform="translate({logoX}, {logoY}) scale({logoScale})" fill="#FFFFFF" opacity="0.9"/>
					
					<!-- Base (keyboard) -->
					{@const baseY = y + screenHeight}
					{@const baseTopX = node.x - screenWidth / 2}
					{@const baseBottomX = node.x - node.iconWidth / 2}
					<path 
						d="M {baseTopX} {baseY} L {baseTopX + screenWidth} {baseY} L {baseBottomX + node.iconWidth} {baseY + baseHeight} L {baseBottomX} {baseY + baseHeight} Z"
						fill="#2c2c2c"
						stroke={node.wireColor}
						stroke-width="1"
					/>
					
					<!-- Keyboard area -->
					<rect x={baseTopX + 6} y={baseY + 3} width={screenWidth - 12} height={baseHeight * 0.55} fill="rgba(0,0,0,0.2)" rx="2"/>
					<!-- Trackpad -->
					<rect x={node.x - screenWidth * 0.2} y={baseY + baseHeight * 0.55 + 5} width={screenWidth * 0.4} height={baseHeight * 0.30} fill="rgba(255,255,255,0.08)" rx="2"/>

				{:else}
					<!-- Unknown device - hexagon -->
					{@const hexRadius = layout().nodeRadius * 0.6}
					{@const hexPoints = Array.from({ length: 6 }, (_, i) => {
						const angle = (i * 60 - 30) * Math.PI / 180;
						return `${node.x + hexRadius * Math.cos(angle)},${node.y + hexRadius * Math.sin(angle)}`;
					}).join(' ')}
					
					<polygon 
						points={hexPoints}
						fill={node.isHighlighted ? 'rgba(255,215,0,0.15)' : 'rgba(255,215,0,0.08)'}
						stroke={node.wireColor}
						stroke-width={node.strokeWidth}
					/>
				{/if}

				<!-- GPU Bar (shown in full and minimized modes) -->
				{#if layout().showFullLabels || isMinimized}
					{@const gpu = getGpuBarDimensions(layout().nodeRadius, node.iconWidth, node.iconHeight)}
					{@const gpuBarX = node.x + gpu.barXOffset}
					{@const gpuBarY = node.y - gpu.gpuBarHeight / 2}
					{@const fillHeight = (node.gpuUsagePercent / 100) * gpu.gpuBarHeight}
					{@const gpuTextFontSize = isMinimized ? Math.max(10, gpu.gpuBarWidth * 0.6) : Math.min(16, Math.max(12, gpu.gpuBarWidth * 0.55))}
					{@const lineSpacing = gpuTextFontSize * 1.25}
					
					<!-- Background -->
					<rect x={gpuBarX} y={gpuBarY} width={gpu.gpuBarWidth} height={gpu.gpuBarHeight} fill="rgba(80, 80, 90, 0.7)" rx="2"/>
					
					<!-- Fill -->
					{#if node.gpuUsagePercent > 0}
						<rect 
							x={gpuBarX}
							y={gpuBarY + (gpu.gpuBarHeight - fillHeight)}
							width={gpu.gpuBarWidth}
							height={fillHeight}
							fill={node.gpuFillColor}
							opacity="0.9"
							rx="2"
						/>
					{/if}
					
					<!-- GPU stats text -->
					<text x={gpuBarX + gpu.gpuBarWidth / 2} y={gpuBarY + gpu.gpuBarHeight / 2 - lineSpacing} text-anchor="middle" dominant-baseline="middle" fill="#FFFFFF" font-size={gpuTextFontSize} font-weight="700" font-family="SF Mono, Monaco, monospace">{node.gpuUsagePercent.toFixed(0)}%</text>
					<text x={gpuBarX + gpu.gpuBarWidth / 2} y={gpuBarY + gpu.gpuBarHeight / 2} text-anchor="middle" dominant-baseline="middle" fill="#FFFFFF" font-size={gpuTextFontSize} font-weight="700" font-family="SF Mono, Monaco, monospace">{!isNaN(node.gpuTemp) ? `${node.gpuTemp.toFixed(0)}°C` : '-'}</text>
					<text x={gpuBarX + gpu.gpuBarWidth / 2} y={gpuBarY + gpu.gpuBarHeight / 2 + lineSpacing} text-anchor="middle" dominant-baseline="middle" fill="#FFFFFF" font-size={gpuTextFontSize} font-weight="700" font-family="SF Mono, Monaco, monospace">{node.sysPower !== null ? `${node.sysPower.toFixed(0)}W` : '-'}</text>
				{/if}

				<!-- Labels -->
				{#if layout().showFullLabels}
					{@const fontSize = getLabelFontSize(layout().nodeRadius, layout().numNodes)}
					<!-- Full mode: Name above, memory below -->
					<text x={node.x} y={node.y - node.iconHeight / 2 - 15} text-anchor="middle" dominant-baseline="middle" fill="#FFD700" font-size={fontSize} font-weight="500" font-family="SF Mono, Monaco, monospace">{node.displayName}</text>
					<text x={node.x} y={node.y + node.iconHeight / 2 + 16} text-anchor="middle" font-size={fontSize * 0.85} font-family="SF Mono, Monaco, monospace">
						<tspan fill="rgba(255,215,0,0.9)">{node.ramUsedFormatted}</tspan><tspan fill="rgba(179,179,179,0.9)">/{node.ramTotalFormatted}</tspan><tspan fill="rgba(179,179,179,0.7)"> ({node.ramUsagePercent.toFixed(0)}%)</tspan>
					</text>
				{:else if layout().showCompactLabels}
					{@const fontSize = getLabelFontSize(layout().nodeRadius, layout().numNodes)}
					<!-- Compact mode: Short name and stats -->
					<text x={node.x} y={node.y + node.iconHeight / 2 + 9} text-anchor="middle" fill="#FFD700" font-size={fontSize} font-family="SF Mono, Monaco, monospace">{node.displayName}</text>
					<text x={node.x} y={node.y + node.iconHeight / 2 + 18} text-anchor="middle" fill="rgba(255,215,0,0.7)" font-size={fontSize * 0.85} font-family="SF Mono, Monaco, monospace">{node.ramUsagePercent.toFixed(0)}%{!isNaN(node.gpuTemp) ? ` ${node.gpuTemp.toFixed(0)}°C` : ''}</text>
				{:else}
					{@const fontSize = getLabelFontSize(layout().nodeRadius, layout().numNodes)}
					<!-- Minimized mode: Name above, memory below -->
					<text x={node.x} y={node.y - node.iconHeight / 2 - 8} text-anchor="middle" fill="#FFD700" font-size={fontSize} font-weight="500" font-family="SF Mono, Monaco, monospace">{node.displayName}</text>
					<text x={node.x} y={node.y + node.iconHeight / 2 + 10} text-anchor="middle" font-size={fontSize * 0.85} font-family="SF Mono, Monaco, monospace">
						<tspan fill="rgba(255,215,0,0.9)">{node.ramUsedFormatted}</tspan><tspan fill="rgba(179,179,179,0.9)">/{node.ramTotalFormatted}</tspan><tspan fill="rgba(179,179,179,0.7)"> ({node.ramUsagePercent.toFixed(0)}%)</tspan>
					</text>
				{/if}
			</g>
		{/each}
	{/if}
</svg>

<style>
	.graph-node {
		transition: transform 0.2s ease, opacity 0.2s ease;
	}
	.graph-node:hover {
		filter: brightness(1.1);
	}
	.graph-link {
		stroke: var(--exo-light-gray, #B3B3B3);
		stroke-width: 1px;
		stroke-dasharray: 4, 4;
		opacity: 0.8;
		animation: flowAnimation 0.75s linear infinite;
	}
	@keyframes flowAnimation {
		from { stroke-dashoffset: 0; }
		to { stroke-dashoffset: -10; }
	}
</style>
