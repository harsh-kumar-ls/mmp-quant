import type { ReactNode } from "react";
import { Info } from "@phosphor-icons/react";
import { cn, fmt } from "@/lib/utils";
import { PenCircle } from "./marks";

/** A sheet of paper. Elevation only where it signals hierarchy. */
export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <section className={cn("paper relative rounded-[20px] p-5 sm:p-7", className)}>{children}</section>;
}

export function CardTitle({ icon, children, sub }: { icon?: ReactNode; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-5">
      <h2 className="flex items-center gap-2.5 text-xl font-bold tracking-tight">
        {icon && <span className="text-coral">{icon}</span>}
        {children}
      </h2>
      {sub && <p className="mt-1.5 max-w-[68ch] text-[14px] leading-relaxed text-mute">{sub}</p>}
    </div>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "ok" | "warn" | "scored" | "neutral"; children: ReactNode }) {
  const tones = {
    ok: "bg-[#dcebdf] text-good", warn: "bg-[#f3e6cc] text-amber",
    scored: "bg-[#f3ddd5] text-coral", neutral: "bg-s2 text-ink-2",
  };
  return <span className={cn("inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold", tones[tone])}>{children}</span>;
}

export function Banner({ title, children, tone = "warn" }: { title: string; children?: ReactNode; tone?: "warn" | "bad" }) {
  return (
    <div role="note" className={cn("flex gap-3 rounded-2xl border-l-4 px-4 py-3.5 text-[14px] leading-relaxed",
      tone === "warn" ? "border-amber bg-[#f6ebd3] text-[#4d3500]" : "border-coral bg-[#f6ddd5] text-[#5c1a0d]")}>
      <Info weight="bold" className="mt-0.5 size-[18px] shrink-0" aria-hidden />
      <div><strong className="mb-0.5 block">{title}</strong>{children}</div>
    </div>
  );
}

export function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-[110px]">
      <dt className="text-[12px] text-mute">{label}</dt>
      <dd className="m-0 mt-0.5 font-mono text-[17px] font-semibold tabular">{children}</dd>
    </div>
  );
}

/** The headline score, circled in red pen like a mark on a script. */
export function ScoreRing({ score, size = 168, label = "of 100" }: { score: number | null; size?: number; label?: string }) {
  return (
    <div className="relative grid shrink-0 place-items-center" style={{ width: size, height: size * 0.62 }}>
      <PenCircle className="absolute inset-0 size-full" />
      <div className="relative text-center">
        <strong className="display block text-[46px] leading-none font-extrabold tracking-tight tabular" data-testid="ring-value">
          {score === null ? "N/A" : fmt(score, 1)}
        </strong>
        <span className="text-xs text-mute">{label}</span>
      </div>
    </div>
  );
}

/** Skeleton, shaped like what is loading — not a spinner. */
export function Spinner({ label }: { label: string }) {
  return (
    <div className="space-y-4 py-6" aria-live="polite" aria-label={label}>
      <div className="h-9 w-2/5 animate-pulse rounded-lg bg-s2" />
      <div className="h-40 animate-pulse rounded-[20px] bg-s2/70" />
      <div className="h-64 animate-pulse rounded-[20px] bg-s2/50" />
      <span className="sr-only">{label}</span>
    </div>
  );
}
