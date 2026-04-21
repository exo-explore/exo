# Dashboard Component

## Overview

The `dashboard/` directory is a SvelteKit 5 single-page app that serves as the built-in web UI for an exo cluster. It provides the cluster topology view, chat interface (text + image generation), model picker, download manager, and trace viewer. It loads at `http://localhost:52415/` — the same origin as the exo API — because the FastAPI server mounts the pre-built static bundle at `/` ([`src/exo/master/api.py:237-244`](../../../src/exo/master/api.py)).

The app has a single hash-routed HTML entry (`index.html`) served by `@sveltejs/adapter-static`, and it talks to the exo API via fetch calls to relative paths (`/state`, `/v1/chat/completions`, `/instance`, `/models`, `/v1/traces`, etc.) ([`dashboard/src/lib/stores/app.svelte.ts:1289`](../../../dashboard/src/lib/stores/app.svelte.ts), [`:1684`](../../../dashboard/src/lib/stores/app.svelte.ts)).

What it shows:

- **Cluster topology** — interactive D3 graph of nodes, edges, RDMA links, and per-device telemetry (chip, RAM, GPU usage, temperature, power) ([`src/lib/stores/app.svelte.ts:28-53`](../../../dashboard/src/lib/stores/app.svelte.ts), [`src/lib/components/TopologyGraph.svelte`](../../../dashboard/src/lib/components/TopologyGraph.svelte)).
- **Chat** — OpenAI-compatible chat completions with streaming, tool calls, markdown+KaTeX rendering, prefill progress, token heatmap, and attachments ([`src/lib/components/ChatForm.svelte`](../../../dashboard/src/lib/components/ChatForm.svelte), [`ChatMessages.svelte`](../../../dashboard/src/lib/components/ChatMessages.svelte), [`TokenHeatmap.svelte`](../../../dashboard/src/lib/components/TokenHeatmap.svelte)).
- **Image generation / edits** — forwards to `/v1/images/generations` and `/v1/images/edits` with partial-image streaming ([`src/lib/stores/app.svelte.ts:2677`](../../../dashboard/src/lib/stores/app.svelte.ts), [`:2951`](../../../dashboard/src/lib/stores/app.svelte.ts)).
- **Model picker** — HuggingFace search, favorites, recents, family grouping, custom model registration ([`src/lib/components/ModelPickerModal.svelte:256-297`](../../../dashboard/src/lib/components/ModelPickerModal.svelte)).
- **Model loading / placement** — previews shard assignments for a given model+sharding+min-nodes, then launches an `Instance` ([`src/lib/stores/app.svelte.ts:702-712`](../../../dashboard/src/lib/stores/app.svelte.ts)).
- **Downloads page** — per-node shard download progress, speed, ETA ([`src/routes/downloads/+page.svelte:1-40`](../../../dashboard/src/routes/downloads/+page.svelte)).
- **Traces** — list and raw-JSON view of per-task inference traces ([`src/routes/traces/+page.svelte:1-40`](../../../dashboard/src/routes/traces/+page.svelte), [`src/routes/traces/[taskId]/+page.svelte`](../../../dashboard/src/routes/traces/%5BtaskId%5D/+page.svelte)).

## Stack

| Tool | Version | Role |
|------|---------|------|
| Svelte | `^5.0.0` | Component framework; runes (`$state`, `$derived`, `$props`) [`dashboard/package.json:23`](../../../dashboard/package.json) |
| SvelteKit | `^2.48.4` | Router + build system [`dashboard/package.json:17`](../../../dashboard/package.json) |
| `@sveltejs/adapter-static` | `^3.0.10` | Builds to fully-static HTML/JS (no Node server) [`dashboard/package.json:16`](../../../dashboard/package.json), [`svelte.config.js:13-19`](../../../dashboard/svelte.config.js) |
| TypeScript | `^5.0.0` | Strict mode enabled [`dashboard/package.json:27`](../../../dashboard/package.json), [`dashboard/tsconfig.json:11`](../../../dashboard/tsconfig.json) |
| Vite | `^6.0.0` | Dev server / bundler [`dashboard/package.json:28`](../../../dashboard/package.json) |
| Tailwind CSS | `^4.0.0` via `@tailwindcss/vite` | Styling; no `tailwind.config` — Tailwind v4 reads CSS-first config [`dashboard/package.json:19,25`](../../../dashboard/package.json), [`vite.config.ts:1,6`](../../../dashboard/vite.config.ts) |
| `d3` | `^7.9.0` | Topology graph force-directed layout [`dashboard/package.json:22`](../../../dashboard/package.json) |
| `marked` + `highlight.js` + `katex` | — | Markdown / code / math rendering in chat messages [`dashboard/package.json:31-33`](../../../dashboard/package.json) |
| `mode-watcher` | `^1.1.0` | Dark/light mode switching [`dashboard/package.json:34`](../../../dashboard/package.json) |

SvelteKit is configured with **hash-based routing** (`router: { type: 'hash' }`) so the app works as a plain static bundle mounted at any prefix without server-side rewrites ([`svelte.config.js:12`](../../../dashboard/svelte.config.js)). The adapter is set with `fallback: 'index.html'` + `strict: true` ([`svelte.config.js:13-19`](../../../dashboard/svelte.config.js)).

## File layout

```
dashboard/
  package.json              scripts + deps (dashboard/package.json:6-36)
  svelte.config.js          adapter-static + hash router (dashboard/svelte.config.js:1-27)
  vite.config.ts            dev proxy to :52415 (dashboard/vite.config.ts:1-15)
  tsconfig.json             strict TS, extends .svelte-kit (dashboard/tsconfig.json:1-14)
  dashboard.nix             dream2nix build for `nix build .#dashboard` (dashboard/dashboard.nix:1-60)
  build/                    output of `npm run build` (served by FastAPI)
  static/                   exo-logo.png, favicon.ico (dashboard/static/)
  src/
    app.html                HTML shell (dashboard/src/app.html:1-13)
    app.css                 Tailwind base + custom tokens
    app.d.ts                SvelteKit type shims
    routes/
      +layout.svelte        App chrome: ConnectionBanner + ToastContainer (dashboard/src/routes/+layout.svelte:1-19)
      +page.svelte          Main app (topology + chat + model picker) — 6,736 lines
      downloads/+page.svelte  Per-node download dashboard (846 lines)
      traces/+page.svelte     Trace list (277 lines)
      traces/[taskId]/+page.svelte  Trace detail (367 lines)
    lib/
      components/           ~22 .svelte files (see below)
      stores/
        app.svelte.ts       central runes-based store (AppStore class, ~3200+ lines)
        favorites.svelte.ts recents.svelte.ts toast.svelte.ts
      types/files.ts        attachment types
      utils/downloads.ts    tagged-union unwrapping for /state download entries
```

**Component inventory** (`dashboard/src/lib/components/`) — exported via [`index.ts`](../../../dashboard/src/lib/components/index.ts):

Chat: `ChatForm`, `ChatMessages`, `ChatAttachments`, `ChatSidebar`, `ChatModelSelector`, `MarkdownContent`, `TokenHeatmap`, `PrefillProgressBar`, `ImageLightbox`, `ImageParamsPanel`.
Model picker: `ModelPickerModal`, `ModelPickerGroup`, `ModelCard`, `ModelFilterPopover`, `HuggingFaceResultItem`, `FamilyLogos`, `FamilySidebar`.
Cluster UI: `TopologyGraph`, `DeviceIcon`, `HeaderNav`, `ConnectionBanner`, `ToastContainer`.

## API integration

All backend calls are same-origin fetches to the exo API served at port 52415. There is **no WebSocket and no SSE channel** for state updates; instead the store polls `/state` once per second ([`src/lib/stores/app.svelte.ts:1274-1277`](../../../dashboard/src/lib/stores/app.svelte.ts)). Streaming chat/image responses use HTTP chunked streams with `stream: true` in the body ([`src/lib/stores/app.svelte.ts:1684`](../../../dashboard/src/lib/stores/app.svelte.ts), [`:355`](../../../dashboard/src/lib/stores/app.svelte.ts)).

| Endpoint | Method | Caller | Purpose |
|----------|--------|--------|---------|
| `/state` | GET | `app.svelte.ts:1289` | Topology + downloads + disk + per-node identity/memory/system, polled every 1 s |
| `/node_id` | GET | `+page.svelte:1246` | Identify current device in topology |
| `/onboarding` | GET / POST | `+page.svelte:673,1264` | First-run onboarding flag |
| `/models` | GET | `+page.svelte:1284` | Built-in model catalog |
| `/models/add` | POST | `+page.svelte:1301` | Register a custom model |
| `/models/custom/{id}` | DELETE | `+page.svelte:1323-1324` | Remove custom model |
| `/models/search?query=…` | GET | `ModelPickerModal.svelte:256,278,297` | HuggingFace search proxy |
| `/instance` | POST / DELETE | `+page.svelte:712,1357,1965,2780,2916` | Launch / tear down a model `Instance` |
| `/instance/{id}` | DELETE | `+page.svelte:1965` | Unload specific instance |
| `/instance/placement` | GET | `+page.svelte:702` | Dry-run placement preview |
| `/instance/previews?model_id=…` | GET | `+page.svelte:2757,2897` | Cached placement previews (polled every 15 s while open, [`app.svelte.ts:1388`](../../../dashboard/src/lib/stores/app.svelte.ts)) |
| `/place_instance` | POST | `+page.svelte:1364` | Alternate placement call path |
| `/v1/chat/completions` | POST | `app.svelte.ts:1684,1891,2368` | OpenAI-compatible streaming chat |
| `/v1/images/generations` | POST | `app.svelte.ts:2677` | Image generation (streaming partials) |
| `/v1/images/edits` | POST | `app.svelte.ts:2951` | Image edit (multipart) |
| `/download/start` | POST | `app.svelte.ts:3108` | Kick off a model download |
| `/v1/traces` | GET / DELETE | `app.svelte.ts:3155,3193` | List / bulk-delete traces |
| `/v1/traces/{taskId}` | GET | `app.svelte.ts:3167,3178`; `traces/+page.svelte:77,89`; `traces/[taskId]/+page.svelte:79,93` | Raw task trace JSON |

The central `AppStore` class (`app.svelte.ts`) holds `$state`-backed topology, instances, downloads, conversations, and a `consecutiveFailures` counter that trips `isConnected=false` after 3 missed `/state` polls (rendered by `ConnectionBanner`) ([`src/lib/stores/app.svelte.ts:595-601`](../../../dashboard/src/lib/stores/app.svelte.ts)).

## Building and running

**Dev mode (standalone):**

```bash
cd dashboard
npm install
npm run dev   # → vite dev, default :5173
```

`vite.config.ts` proxies `/v1`, `/state`, `/models`, `/instance` from the dev server to `http://localhost:52415` so the dev UI talks to a locally-running exo ([`vite.config.ts:7-14`](../../../dashboard/vite.config.ts)). Note: `/download`, `/node_id`, `/onboarding`, `/v1/traces` are **not in the proxy list** — see Gotchas.

**Production build:**

```bash
cd dashboard && npm install && npm run build
```

Output lands in `dashboard/build/` with `index.html`, `_app/immutable/`, `env.js`, `version.json`, and copied static assets (`exo-logo.png`, `favicon.ico`). This matches `svelte.config.js`'s `adapter({ pages: 'build', assets: 'build', fallback: 'index.html' })` ([`svelte.config.js:13-19`](../../../dashboard/svelte.config.js)).

**Commands available:**

| Script | Command | [`dashboard/package.json:7-11`](../../../dashboard/package.json) |
|--------|---------|---|
| `dev` | `vite dev` | line 7 |
| `build` | `vite build` | line 8 |
| `preview` | `vite preview` | line 9 |
| `prepare` | `svelte-kit sync` | line 10 |
| `check` | `svelte-kit sync && svelte-check --tsconfig ./tsconfig.json` | line 11 |

The repo root [`README.md:117-118`](../../../README.md) instructs `cd exo/dashboard && npm install && npm run build && cd ..` as a prerequisite before `uv run exo`.

## Packaging with exo

The dashboard is **not** bundled into the Python package as a resource. Instead, exo locates the already-built `dashboard/build/` directory at runtime:

- `find_dashboard()` walks up from `src/exo/utils/dashboard_path.py` looking for a sibling `dashboard/build/` directory that contains `index.html` ([`src/exo/utils/dashboard_path.py:34-49`](../../../src/exo/utils/dashboard_path.py)).
- If that fails (e.g., running from a PyInstaller bundle), it falls back to `sys._MEIPASS/dashboard/` ([`src/exo/utils/dashboard_path.py:52-59`](../../../src/exo/utils/dashboard_path.py)).
- If neither exists, it raises: `"Unable to locate dashboard assets - you probably forgot to run cd dashboard && npm install && npm run build && cd .."` ([`src/exo/utils/dashboard_path.py:37-40`](../../../src/exo/utils/dashboard_path.py)).
- The resolved path becomes `DASHBOARD_DIR` ([`src/exo/shared/constants.py:66-69`](../../../src/exo/shared/constants.py)); the env var `EXO_DASHBOARD_DIR` overrides it (resolved relative to `$HOME`).

The FastAPI master **mounts the build directory as static files at `/`** with `html=True` so `index.html` is served for the root and deep links fall through to SPA routing ([`src/exo/master/api.py:237-244`](../../../src/exo/master/api.py)):

```python
self.app.mount(
    "/",
    StaticFiles(directory=DASHBOARD_DIR, html=True),
    name="dashboard",
)
```

The mount is attached **after** all API routes are registered via `_setup_routes()` ([`src/exo/master/api.py:233-244`](../../../src/exo/master/api.py)), so API paths take precedence and unmatched paths fall through to the SPA.

A separate Nix derivation in [`dashboard/dashboard.nix:1-60`](../../../dashboard/dashboard.nix) builds the dashboard via dream2nix from `package-lock.json` and installs `build/` to `$out/build` — used by `nix build .#dashboard` and referenced from the top-level flake so `nix run .#exo` works without a manual `npm run build`.

## Gotchas

- **Build is mandatory before `uv run exo`.** The error message is explicit ([`src/exo/utils/dashboard_path.py:37-40`](../../../src/exo/utils/dashboard_path.py)). CI / fresh clones forget this frequently — the `README.md` quick-start lists it as a prerequisite ([`README.md:117-118`](../../../README.md)).
- **Hash routing.** The app uses `router: { type: 'hash' }` ([`svelte.config.js:12`](../../../dashboard/svelte.config.js)), so URLs look like `http://localhost:52415/#/traces`. Direct links to `/traces` **without** the hash work only because FastAPI's `StaticFiles(html=True)` serves `index.html` as a fallback, and the SvelteKit bundle then reads the hash client-side.
- **`/state` polling is 1 Hz unconditionally** ([`src/lib/stores/app.svelte.ts:1276`](../../../dashboard/src/lib/stores/app.svelte.ts)). There's no backoff when the tab is hidden. After 3 consecutive failures, `ConnectionBanner` appears ([`src/lib/stores/app.svelte.ts:597-598`](../../../dashboard/src/lib/stores/app.svelte.ts)).
- **Dev proxy is incomplete.** `vite.config.ts` proxies only `/v1`, `/state`, `/models`, `/instance` ([`vite.config.ts:8-13`](../../../dashboard/vite.config.ts)). Calls to `/node_id`, `/onboarding`, `/download/start`, `/v1/traces`, `/place_instance` from `npm run dev` will 404 unless you add them to the proxy or hit the production build instead.
- **Huge single-page route.** `src/routes/+page.svelte` is **6,736 lines** and hosts topology + chat + model picker + instance management in one file. Refactoring is non-trivial but the component boundary is natural for splitting.
- **Tailwind v4 has no `tailwind.config.{js,ts}`** — config lives in CSS via `@theme` / `@config` directives and the `@tailwindcss/vite` plugin ([`vite.config.ts:1,6`](../../../dashboard/vite.config.ts)). Don't look for a JS config file.
- **`paths: { relative: true }`** in svelte.config ([`svelte.config.js:9-11`](../../../dashboard/svelte.config.js)) means asset URLs in `index.html` are relative, so the bundle can be mounted at any prefix. Combined with `html=True` on the FastAPI mount, sub-paths also serve `index.html`.
- **No `.env` / `PUBLIC_*` config.** Backend URL is hard-coded to same-origin relative paths. To point the UI at a remote exo, use a reverse proxy or the Vite dev proxy.
- **`EXO_DASHBOARD_DIR` is resolved relative to `$HOME`**, not the CWD ([`src/exo/shared/constants.py:67-69`](../../../src/exo/shared/constants.py)) — surprising if you set it to an absolute path expecting it to be used verbatim.

## See also

- [`../workflows/running-exo.md`](../workflows/running-exo.md) — end-to-end run instructions including dashboard build step
- [`../architecture/data-flow.md`](../architecture/data-flow.md) — how `/state` and events propagate from Master to the UI
- [`./master.md`](./master.md) — the API server that mounts and serves this bundle

---

**Sources**

- `dashboard/package.json` — script definitions and dependency versions
- `dashboard/svelte.config.js` — adapter-static + hash router + aliases
- `dashboard/vite.config.ts` — dev server + proxy
- `dashboard/tsconfig.json` — strict TypeScript
- `dashboard/dashboard.nix` — Nix build via dream2nix
- `dashboard/src/app.html` — HTML shell
- `dashboard/src/routes/+layout.svelte` — app chrome
- `dashboard/src/routes/+page.svelte` — main UI (6,736 lines)
- `dashboard/src/routes/downloads/+page.svelte`, `traces/+page.svelte`, `traces/[taskId]/+page.svelte` — secondary routes
- `dashboard/src/lib/stores/app.svelte.ts` — central AppStore + all fetch calls
- `dashboard/src/lib/components/index.ts` — public component exports
- `dashboard/src/lib/utils/downloads.ts` — state parsing helpers
- `src/exo/utils/dashboard_path.py` — dashboard asset discovery
- `src/exo/shared/constants.py` — `DASHBOARD_DIR`
- `src/exo/master/api.py` — FastAPI `StaticFiles` mount
- `README.md` — build+run quick start

**Last Indexed:** 2026-04-21 | exo commit: c0d5bf92
