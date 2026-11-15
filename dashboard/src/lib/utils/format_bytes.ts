const UNITS = ["B", "KB", "MB", "GB", "TB"];

/**
 * Formats a byte count with binary units, e.g. 1536 -> "1.5 KB".
 *
 * The unit is picked after rounding, so a value that rounds up to 1024 moves
 * to the next unit: 1048575 bytes is "1 MB", not "1024 KB".
 */
export function formatBytes(
  bytes: number,
  decimals = 1,
  separator = " ",
): string {
  if (!bytes || bytes <= 0) return `0${separator}B`;
  let i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    UNITS.length - 1,
  );
  let value = parseFloat((bytes / Math.pow(1024, i)).toFixed(decimals));
  if (value >= 1024 && i < UNITS.length - 1) {
    i++;
    value = parseFloat((bytes / Math.pow(1024, i)).toFixed(decimals));
  }
  return `${value}${separator}${UNITS[i]}`;
}
