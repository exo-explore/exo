import { browser } from "$app/environment";

let _isLight = $state(false);

export const theme = {
  get isLight() {
    return _isLight;
  },

  init() {
    if (!browser) return;
    _isLight = document.documentElement.classList.contains("light");
  },

  toggle() {
    if (!browser) return;
    _isLight = !_isLight;
    if (_isLight) {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
      localStorage.setItem("exo-theme", "light");
    } else {
      document.documentElement.classList.remove("light");
      document.documentElement.classList.add("dark");
      localStorage.setItem("exo-theme", "dark");
    }
  },
};
