/* Hand-drawn red-pen marks. Paths are deliberately imperfect (overshoot,
   uneven pressure) — the point is that a person graded this. */
import { cn } from "@/lib/utils";

export function PenCircle({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 220 120" className={cn("pointer-events-none", className)} aria-hidden fill="none">
      <path className="draw" style={{ ["--len" as string]: 640 }}
        d="M150 14C112 4 46 10 22 38 2 62 18 98 70 108c44 9 104 2 128-24 20-22 6-56-32-66-30-8-70-6-96 2"
        stroke="var(--color-coral)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function PenUnderline({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 300 24" preserveAspectRatio="none" className={cn("pointer-events-none", className)} aria-hidden fill="none">
      <path className="draw" style={{ ["--len" as string]: 360 }}
        d="M4 15c40-6 92-9 150-8 50 1 98 4 142 9M30 20c50-4 120-6 200-3"
        stroke="var(--color-coral)" strokeWidth="4.5" strokeLinecap="round" />
    </svg>
  );
}

export function PenTick({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 32" className={cn("pointer-events-none", className)} aria-hidden fill="none">
      <path className="draw" style={{ ["--len" as string]: 80 }} d="M4 17c5 5 9 9 12 12C22 18 30 8 37 3"
        stroke="var(--color-coral)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function PenArrow({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 120 70" className={cn("pointer-events-none", className)} aria-hidden fill="none">
      <path className="draw" style={{ ["--len" as string]: 220 }} d="M6 8c20 30 50 48 98 50M88 44c8 6 13 11 17 15-7 3-14 5-21 9"
        stroke="var(--color-coral)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
