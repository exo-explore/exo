/**
 * RecentsStore - Manages recently launched models with localStorage persistence
 */

import { browser } from "$app/environment";

const RECENTS_KEY = "exo-recent-models";
const MAX_RECENT_MODELS = 20;

interface RecentEntry {
  modelId: string;
  launchedAt: number;
}

class RecentsStore {
  recents = $state<RecentEntry[]>([]);

  constructor() {
    if (browser) {
      this.loadFromStorage();
    }
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem(RECENTS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as RecentEntry[];
        this.recents = parsed;
      }
    } catch (error) {
      console.error("Failed to load recent models:", error);
    }
  }

  private saveToStorage() {
    try {
      localStorage.setItem(RECENTS_KEY, JSON.stringify(this.recents));
    } catch (error) {
      console.error("Failed to save recent models:", error);
    }
  }

  recordLaunch(modelId: string) {
    // Remove existing entry for this model (if any) to move it to top
    const filtered = this.recents.filter((r) => r.modelId !== modelId);
    // Prepend new entry
    const next = [{ modelId, launchedAt: Date.now() }, ...filtered];
    // Cap at max
    this.recents = next.slice(0, MAX_RECENT_MODELS);
    this.saveToStorage();
  }

  getRecentModelIds(): string[] {
    return this.recents.map((r) => r.modelId);
  }

  hasAny(): boolean {
    return this.recents.length > 0;
  }

  clearAll() {
    this.recents = [];
    this.saveToStorage();
  }
}

export const recentsStore = new RecentsStore();

export const hasRecents = () => recentsStore.hasAny();
export const getRecentModelIds = () => recentsStore.getRecentModelIds();
export const getRecentEntries = () => recentsStore.recents;
export const recordRecentLaunch = (modelId: string) =>
  recentsStore.recordLaunch(modelId);
export const clearRecents = () => recentsStore.clearAll();
