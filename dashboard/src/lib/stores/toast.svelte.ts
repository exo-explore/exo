/**
 * Toast notification store - Global notification system for the EXO dashboard.
 *
 * Usage:
 *   import { addToast, dismissToast, toasts } from "$lib/stores/toast.svelte";
 *   addToast({ type: "success", message: "Model launched" });
 *   addToast({ type: "error", message: "Connection lost", persistent: true });
 */

type ToastType = "success" | "error" | "warning" | "info";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  /** Auto-dismiss after this many ms. 0 = persistent (must be dismissed manually). */
  duration: number;
  createdAt: number;
}

interface ToastInput {
  type: ToastType;
  message: string;
  /** If true, toast stays until manually dismissed. Default: false. */
  persistent?: boolean;
  /** Auto-dismiss duration in ms. Default: 4000 for success/info, 6000 for error/warning. */
  duration?: number;
}

const DEFAULT_DURATIONS: Record<ToastType, number> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 6000,
};

let toastList = $state<Toast[]>([]);
const timers = new Map<string, ReturnType<typeof setTimeout>>();

function generateId(): string {
  return `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function addToast(input: ToastInput): string {
  const id = generateId();
  const duration = input.persistent
    ? 0
    : (input.duration ?? DEFAULT_DURATIONS[input.type]);

  const toast: Toast = {
    id,
    type: input.type,
    message: input.message,
    duration,
    createdAt: Date.now(),
  };

  toastList = [...toastList, toast];

  if (duration > 0) {
    const timer = setTimeout(() => dismissToast(id), duration);
    timers.set(id, timer);
  }

  return id;
}

export function dismissToast(id: string): void {
  const timer = timers.get(id);
  if (timer) {
    clearTimeout(timer);
    timers.delete(id);
  }
  toastList = toastList.filter((t) => t.id !== id);
}

/** Dismiss all toasts matching a message (useful for dedup). */
export function dismissByMessage(message: string): void {
  const matching = toastList.filter((t) => t.message === message);
  for (const t of matching) {
    dismissToast(t.id);
  }
}

export function toasts(): Toast[] {
  return toastList;
}
