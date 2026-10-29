/**
 * EXO yellow at the given opacity, for places that take a colour string:
 * SVG attributes, d3 and inline styles. It refers to the --exo-yellow token
 * in app.css, which holds the one value (the logo's #FFDE00); CSS and
 * Tailwind classes (bg-exo-yellow/50 and the like) use the token directly.
 */
export function exoYellow(opacity = 1): string {
  if (opacity >= 1) return "var(--exo-yellow)";
  const percent = Math.round(opacity * 1000) / 10;
  return `color-mix(in srgb, var(--exo-yellow) ${percent}%, transparent)`;
}
