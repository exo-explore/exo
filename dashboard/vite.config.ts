import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		proxy: {
			'/v1': 'http://localhost:8000',
			'/state': 'http://localhost:8000',
			'/models': 'http://localhost:8000',
			'/instance': 'http://localhost:8000'
		}
	}
});

