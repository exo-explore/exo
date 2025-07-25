import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    port: 3000,
    host: true,
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022'
  },
  optimizeDeps: {
    exclude: ['libp2p']
  },
  define: {
    global: 'globalThis'
  }
});
