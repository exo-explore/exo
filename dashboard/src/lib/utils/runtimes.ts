export const RUNTIME_LABELS = {
  MlxRing: "Standard (TCP/IP)",
  MlxJaccl: "RDMA over Thunderbolt",
  TensorFold: "TensorFold",
} as const;

export type InstanceMeta = keyof typeof RUNTIME_LABELS;
type Sharding = "Pipeline" | "Tensor";

export const RUNTIMES: InstanceMeta[] = ["MlxRing", "MlxJaccl", "TensorFold"];

export const RUNTIME_DESCRIPTIONS: Record<InstanceMeta, string> = {
  MlxRing: "MLX over Wi-Fi, Ethernet or Thunderbolt.",
  MlxJaccl: "MLX with direct memory access between connected Macs.",
  TensorFold: "Native Metal or CUDA inference on one device.",
};

export function isInstanceMeta(value: unknown): value is InstanceMeta {
  return typeof value === "string" && Object.hasOwn(RUNTIME_LABELS, value);
}

export function instanceRuntimeLabel(tag: string | null): string | null {
  if (tag === "MlxRingInstance") return RUNTIME_LABELS.MlxRing;
  if (tag === "MlxJacclInstance") return RUNTIME_LABELS.MlxJaccl;
  if (tag === "TensorFoldInstance") return RUNTIME_LABELS.TensorFold;
  return null;
}

export function alternateRuntimeSettings(
  selected: InstanceMeta,
  sharding: Sharding,
): Array<{ sharding: Sharding; runtime: InstanceMeta }> {
  const otherSharding = sharding === "Pipeline" ? "Tensor" : "Pipeline";
  const runtimes = [
    ...RUNTIMES.filter((runtime) => runtime !== selected),
    selected,
  ];
  const shardings: Sharding[] = [sharding, otherSharding];
  return shardings.flatMap((split) =>
    runtimes
      .filter((runtime) => !(runtime === selected && split === sharding))
      .filter((runtime) => runtime !== "TensorFold" || split === "Pipeline")
      .map((runtime) => ({
        sharding: split,
        runtime,
      })),
  );
}
