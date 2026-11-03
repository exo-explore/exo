/**
 * Whether to suggest enabling RDMA.
 *
 * RDMA needs devices connected to each other with Thunderbolt 5 cables, and
 * rdma_ctl enabled on each. Every recent Mac has Thunderbolt ports, so
 * having the ports says nothing: a cluster on Ethernet or Wi-Fi has nothing
 * to enable. What tells a live link from an idle port is its speed, as the
 * device reports it from system_profiler: an idle Thunderbolt 5 port says
 * "Up to 120 Gb/s", a connected one the speed of its link ("80 Gb/s").
 */

export interface ThunderboltInterface {
  linkSpeed?: string;
}

/** Thunderbolt 5 links run at 80 Gb/s or more (Thunderbolt 4: 40). */
const THUNDERBOLT_5_GBPS = 80;

/** Whether a port has a live Thunderbolt 5 link (not just the hardware). */
export function isThunderbolt5Link(iface: ThunderboltInterface): boolean {
  const speed = (iface.linkSpeed ?? "").trim();
  if (!speed || /^up to/i.test(speed)) return false;
  const gbps = Number.parseFloat(speed);
  return Number.isFinite(gbps) && gbps >= THUNDERBOLT_5_GBPS;
}

/**
 * The devices connected with Thunderbolt 5 that don't have RDMA enabled,
 * when at least two devices are connected that way; empty otherwise.
 */
export function devicesMissingRdma(
  thunderbolt: Record<string, { interfaces: ThunderboltInterface[] }>,
  rdmaCtl: Record<string, { enabled: boolean }>,
): string[] {
  const linked = Object.entries(thunderbolt)
    .filter(([, node]) => node.interfaces.some(isThunderbolt5Link))
    .map(([nodeId]) => nodeId);
  if (linked.length < 2) return [];
  return linked.filter((nodeId) => rdmaCtl[nodeId]?.enabled !== true);
}
