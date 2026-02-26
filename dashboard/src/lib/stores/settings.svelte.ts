/**
 * SettingsStore - Manages exo runtime settings via the /settings API.
 */

export interface MemorySettings {
  oom_prevention: boolean;
  memory_threshold: number;
  memory_floor_gb: number;
}

export interface GenerationSettings {
  prefill_step_size: number;
  max_tokens: number;
  kv_cache_bits: 4 | 8 | null;
}

export interface ExoSettings {
  memory: MemorySettings;
  generation: GenerationSettings;
}

function defaultSettings(): ExoSettings {
  return {
    memory: {
      oom_prevention: false,
      memory_threshold: 0.8,
      memory_floor_gb: 5.0,
    },
    generation: {
      prefill_step_size: 4096,
      max_tokens: 32168,
      kv_cache_bits: null,
    },
  };
}

class SettingsStore {
  settings = $state<ExoSettings>(defaultSettings());
  loading = $state(false);
  error = $state<string | null>(null);

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const response = await fetch("/settings");
      if (!response.ok) {
        throw new Error(`Failed to fetch settings: ${response.status}`);
      }
      this.settings = (await response.json()) as ExoSettings;
    } catch (err) {
      console.error("Failed to load settings:", err);
      this.error = err instanceof Error ? err.message : "Unknown error";
    } finally {
      this.loading = false;
    }
  }

  async save(updated: ExoSettings): Promise<boolean> {
    this.loading = true;
    this.error = null;
    try {
      const response = await fetch("/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updated),
      });
      if (!response.ok) {
        throw new Error(`Failed to save settings: ${response.status}`);
      }
      this.settings = (await response.json()) as ExoSettings;
      return true;
    } catch (err) {
      console.error("Failed to save settings:", err);
      this.error = err instanceof Error ? err.message : "Unknown error";
      return false;
    } finally {
      this.loading = false;
    }
  }

  resetToDefaults(): ExoSettings {
    return defaultSettings();
  }
}

export const settingsStore = new SettingsStore();
