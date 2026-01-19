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

	/**
	 * Add a model to favorites
	 */
	add(baseModelId: string) {
		const next = new Set(this.favorites);
		next.add(baseModelId);
		this.favorites = next;
		this.saveToStorage();
	}

	/**
	 * Remove a model from favorites
	 */
	remove(baseModelId: string) {
		const next = new Set(this.favorites);
		next.delete(baseModelId);
		this.favorites = next;
		this.saveToStorage();
	}

	/**
	 * Toggle favorite status of a model
	 */
	toggle(baseModelId: string) {
		if (this.favorites.has(baseModelId)) {
			this.remove(baseModelId);
		} else {
			this.add(baseModelId);
		}
	}

	/**
	 * Check if a model is a favorite
	 */
	isFavorite(baseModelId: string): boolean {
		return this.favorites.has(baseModelId);
	}

	/**
	 * Get all favorite model IDs
	 */
	getAll(): string[] {
		return Array.from(this.favorites);
	}

	/**
	 * Get favorites as a Set (for efficient lookup)
	 */
	getSet(): Set<string> {
		return new Set(this.favorites);
	}

	/**
	 * Check if there are any favorites
	 */
	hasAny(): boolean {
		return this.favorites.size > 0;
	}

	/**
	 * Clear all favorites
	 */
	clearAll() {
		this.favorites = new Set();
		this.saveToStorage();
	}
}

export const favoritesStore = new FavoritesStore();

// Reactive exports
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
