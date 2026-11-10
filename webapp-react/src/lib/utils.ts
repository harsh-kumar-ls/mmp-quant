import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** The shadcn / 21st.dev class helper, so catalogue components drop in as-is. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const fmt = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined ? "N/A" : Number(x).toFixed(d);
export const pct = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;
export const timecode = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
/** m:ss.s — the precision timing decisions are made at (0.5 s frames). */
export const tc = (t: number) => {
  const m = Math.floor(t / 60);
  return `${m}:${(t - 60 * m).toFixed(1).padStart(4, "0")}`;
};

/* Score-bar colour is magnitude: ONE hue (blue), darker = more (light theme).
   Only the ramp steps that clear 3:1 against the bar track (#e4dccd) are used:
   3.2, 4.3, 5.8, 7.7 : 1. The value is always printed beside the bar. */
const SEQ = ["#3f79cc", "#2d64b3", "#22518f", "#1b3f70"];
export const seqColor = (score: number | null | undefined) =>
  SEQ[Math.max(0, Math.min(SEQ.length - 1, Math.round(((score || 0) / 100) * (SEQ.length - 1))))];
