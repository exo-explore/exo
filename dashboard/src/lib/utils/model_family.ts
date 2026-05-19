// Mirrors src/exo/shared/models/model_cards.py:derive_base_model
const QUANT_SUFFIXES = new RegExp(
  "[-_ ](?:MLX|MXFP[0-9]+|NVFP[0-9]+|GPTQ|AWQ|GGUF|fp16|bf16|fp8|int[0-9]+|[0-9]+(?:\\.[0-9]+)?bit|Q[0-9]+(?:_[A-Z0-9]+)?|gs[0-9]+)" +
    "(?:[-_ ](?:MLX|Q[0-9]+|Int[0-9]+|[A-Z0-9]+|gs[0-9]+))*$",
  "i",
);

function normalize(s: string): string {
  return s
    .replaceAll("-", " ")
    .replaceAll("_", " ")
    .replaceAll("  ", " ")
    .trim();
}

export function deriveBaseModel(modelId: string): string {
  const short = modelId.includes("/")
    ? (modelId.split("/").pop() ?? modelId)
    : modelId;
  const stripped = short.replace(QUANT_SUFFIXES, "");
  return normalize(stripped);
}

export function baseModelsCompatible(a: string, b: string): boolean {
  return deriveBaseModel(a).toLowerCase() === deriveBaseModel(b).toLowerCase();
}

// Mirrors src/exo/shared/models/model_cards.py:derive_family
export function deriveFamily(modelId: string): string {
  const short = modelId.includes("/")
    ? (modelId.split("/").pop() ?? modelId)
    : modelId;
  const stripped = short
    .replace(QUANT_SUFFIXES, "")
    .toLowerCase()
    .replaceAll("_", "-");
  const parts = stripped.split(/[-.]/);
  const familyParts: string[] = [];
  for (const p of parts) {
    if (/^\d+$/.test(p) || /^\d+[bm]?$/i.test(p)) break;
    familyParts.push(p);
  }
  return familyParts.length > 0 ? familyParts.join("-") : stripped;
}
