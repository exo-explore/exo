import { browser } from "$app/environment";

export type ThemeName = "light" | "dark" | "solar";
export type ThemeMode = ThemeName | "system";

const STORAGE_KEY = "exo-theme";
const VALID_THEMES: ReadonlyArray<ThemeName> = ["light", "dark", "solar"];
const VALID_MODES: ReadonlyArray<ThemeMode> = [...VALID_THEMES, "system"];

function isValidMode(value: string | null): value is ThemeMode {
  return value !== null && (VALID_MODES as readonly string[]).includes(value);
}

function readStoredMode(): ThemeMode {
  if (!browser) return "light";
  const raw = localStorage.getItem(STORAGE_KEY);
  return isValidMode(raw) ? raw : "light";
}

function systemPrefersDark(): boolean {
  if (!browser || !window.matchMedia) return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function resolve(mode: ThemeMode): ThemeName {
  if (mode === "system") return systemPrefersDark() ? "dark" : "light";
  return mode;
}

function createThemeStore() {
  let mode = $state<ThemeMode>(readStoredMode());
  let effective = $state<ThemeName>(resolve(mode));

  function apply() {
    if (!browser) return;
    document.body.setAttribute("data-theme", effective);
  }

  function setMode(next: ThemeMode) {
    mode = next;
    effective = resolve(next);
    if (browser) {
      localStorage.setItem(STORAGE_KEY, next);
      apply();
    }
  }

  function cycle() {
    const order: ThemeMode[] = ["light", "dark", "solar", "system"];
    const idx = order.indexOf(mode);
    const next = order[(idx + 1) % order.length]!;
    setMode(next);
  }

  if (browser) {
    // Re-resolve when the OS preference changes while in "system" mode.
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if (mode === "system") {
        effective = resolve(mode);
        apply();
      }
    };
    mq.addEventListener("change", onChange);
    apply();
  }

  return {
    get mode() {
      return mode;
    },
    get effective() {
      return effective;
    },
    setMode,
    cycle,
  };
}

export const theme = createThemeStore();

export const THEME_OPTIONS: Array<{
  mode: ThemeMode;
  label: string;
  description: string;
}> = [
  { mode: "light", label: "Light", description: "Clean white surfaces, oMLX-feel default." },
  { mode: "dark", label: "Dark", description: "The original new-shell look." },
  { mode: "solar", label: "Solar", description: "Warm paper with chocolate accents." },
  { mode: "system", label: "System", description: "Follows your macOS appearance." },
];
