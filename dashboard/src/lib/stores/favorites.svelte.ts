/**
 * FavoritesStore - Manages favorite models with localStorage persistence
 */

import { browser } from "$app/environment";

const FAVORITES_KEY = "exo-favorite-models";

class FavoritesStore {
  favorites = $state<Set<string>>(new Set());

  constructor() {
    if (browser) {
      this.loadFromStorage();
    }
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem(FAVORITES_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as string[];
        this.favorites = new Set(parsed);
      }
    } catch (error) {
      console.error("Failed to load favorites:", error);
    }
  }

  private saveToStorage() {
    try {
      const array = Array.from(this.favorites);
      localStorage.setItem(FAVORITES_KEY, JSON.stringify(array));
    } catch (error) {
      console.error("Failed to save favorites:", error);
    }
  }

  add(baseModelId: string) {
    const next = new Set(this.favorites);
    next.add(baseModelId);
    this.favorites = next;
    this.saveToStorage();
  }

  remove(baseModelId: string) {
    const next = new Set(this.favorites);
    next.delete(baseModelId);
    this.favorites = next;
    this.saveToStorage();
  }

  toggle(baseModelId: string) {
    if (this.favorites.has(baseModelId)) {
      this.remove(baseModelId);
    } else {
      this.add(baseModelId);
    }
  }

  isFavorite(baseModelId: string): boolean {
    return this.favorites.has(baseModelId);
  }

  getAll(): string[] {
    return Array.from(this.favorites);
  }

  getSet(): Set<string> {
    return new Set(this.favorites);
  }

  hasAny(): boolean {
    return this.favorites.size > 0;
  }

  clearAll() {
    this.favorites = new Set();
    this.saveToStorage();
  }
}

export const favoritesStore = new FavoritesStore();

export const favorites = () => favoritesStore.favorites;
export const hasFavorites = () => favoritesStore.hasAny();
export const isFavorite = (baseModelId: string) =>
  favoritesStore.isFavorite(baseModelId);
export const toggleFavorite = (baseModelId: string) =>
  favoritesStore.toggle(baseModelId);
export const addFavorite = (baseModelId: string) =>
  favoritesStore.add(baseModelId);
export const removeFavorite = (baseModelId: string) =>
  favoritesStore.remove(baseModelId);
export const getFavorites = () => favoritesStore.getAll();
export const getFavoritesSet = () => favoritesStore.getSet();
export const clearFavorites = () => favoritesStore.clearAll();
