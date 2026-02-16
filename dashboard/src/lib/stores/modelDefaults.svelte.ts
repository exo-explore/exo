/**
 * ModelDefaultsStore - Remembers per-model configuration (sharding, instance type, min nodes)
 */

import { browser } from "$app/environment";

const MODEL_CONFIGS_KEY = "exo-model-configs";

type InstanceMeta = "MlxRing" | "MlxIbv" | "MlxJaccl";

interface ModelConfig {
  sharding: "Pipeline" | "Tensor";
  instanceType: InstanceMeta;
  minNodes: number;
}

class ModelDefaultsStore {
  configs = $state<Map<string, ModelConfig>>(new Map());

  constructor() {
    if (browser) {
      this.loadFromStorage();
    }
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem(MODEL_CONFIGS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Array<[string, ModelConfig]>;
        this.configs = new Map(parsed);
      }
    } catch (error) {
      console.error("Failed to load model configs:", error);
    }
  }

  private saveToStorage() {
    try {
      const entries = Array.from(this.configs.entries());
      localStorage.setItem(MODEL_CONFIGS_KEY, JSON.stringify(entries));
    } catch (error) {
      console.error("Failed to save model configs:", error);
    }
  }

  getConfig(baseModelId: string): ModelConfig | undefined {
    return this.configs.get(baseModelId);
  }

  saveConfig(baseModelId: string, config: ModelConfig) {
    const next = new Map(this.configs);
    next.set(baseModelId, config);
    this.configs = next;
    this.saveToStorage();
  }
}

export const modelDefaultsStore = new ModelDefaultsStore();

export const getModelConfig = (baseModelId: string) =>
  modelDefaultsStore.getConfig(baseModelId);
export const saveModelConfig = (baseModelId: string, config: ModelConfig) =>
  modelDefaultsStore.saveConfig(baseModelId, config);
