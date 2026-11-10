import { useEffect } from "react";
import { CaretLeft, CaretRight, MathOperations, Stack, X } from "@phosphor-icons/react";
import type { Report } from "@/lib/api";
import { cn, fmt } from "@/lib/utils";
import { WindowCalc } from "./WindowCalc";

export type Principle = "redundancy" | "coherence";

const th = "border-b border-line px-2 py-2 text-left text-[10.5px] font-semibold uppercase tracking-wide text-mute";
const td = "border-b border-line px-2 py-2 text-ink-2 tabular";
const f2 = (x: number) => x.toFixed(2);

export function SegmentDrawer({ report, principle, index, onIndex, onClose }: {
  report: Report; principle: Principle; index: number; onIndex: (i: number) => void; onClose: () => void;
}) {
  const segs = principle === "coherence" ? report.coherence!.segments : report.segments;
  const seg = segs[index];
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && index > 0) onIndex(index - 1);
      if (e.key === "ArrowRight" && index < segs.length - 1) onIndex(index + 1);
    };
    document.addEventListener("keydown", key);
    document.documentElement.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", key); document.documentElement.style.overflow = ""; };
  }, [index, segs.length, onClose, onIndex]);

  return (
    <>
      <div className="fixed inset-0 z-40 bg-[rgba(29,26,22,.32)] backdrop-blur-[1.5px]" onClick={onClose} />
      <aside role="dialog" aria-modal="true" data-testid="drawer" data-principle={principle}
        className="slide-in fixed inset-y-0 right-0 z-50 flex w-full max-w-[780px] flex-col bg-bg shadow-[-24px_0_60px_-20px_rgba(92,64,30,.35)]">
        <header className="flex items-center gap-3 border-b border-line px-5 py-4 sm:px-7">
          <div className="min-w-0 flex-1">
            <div className="font-mono text-[12px] text-mute" data-testid="drawer-kicker">
              {principle} · segment {index + 1} of {segs.length}
            </div>
            <div className="display text-[24px] font-extrabold tracking-tight tabular">{seg.timecode}</div>
          </div>
          <div className="text-right"><strong className="display text-[28px] font-extrabold text-coral tabular" data-testid="drawer-score">{fmt(seg.score)}</strong><div className="text-xs text-mute">segment score</div></div>
          <div className="flex gap-1">
            <IconBtn label="Previous segment" disabled={index === 0} onClick={() => onIndex(index - 1)}><CaretLeft weight="bold" className="size-4" /></IconBtn>
            <IconBtn label="Next segment" disabled={index === segs.length - 1} onClick={() => onIndex(index + 1)}><CaretRight weight="bold" className="size-4" /></IconBtn>
            <IconBtn label="Close" onClick={onClose}><X weight="bold" className="size-4" /></IconBtn>
          </div>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5 sm:px-7" data-testid="drawer-body">
          {principle === "redundancy" ? <RedundancySegment report={report} index={index} /> : <CoherenceSegment report={report} index={index} />}
        </div>
      </aside>
    </>
  );
}

function IconBtn({ label, children, onClick, disabled }: { label: string; children: React.ReactNode; onClick: () => void; disabled?: boolean }) {
  return <button type="button" aria-label={label} title={label} disabled={disabled} onClick={onClick}
    className="grid size-9 place-items-center rounded-xl bg-s2 text-ink-2 transition hover:bg-s3 hover:text-ink active:scale-95 disabled:opacity-35">{children}</button>;
}

function SectionTitle({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return <div className="mb-2 mt-1 flex items-center gap-2 text-[16px] font-bold"><span className="text-coral">{icon}</span>{children}</div>;
}

function RedundancySegment({ report, index }: { report: Report; index: number }) {
  const seg = report.segments[index];
  const ws = seg.windows;
  const num = ws.reduce((a, w) => a + (w.qualifies ? (w.score ?? 0) * w.duration : 0), 0);
  const den = ws.reduce((a, w) => a + w.duration, 0);
  const firstScored = ws.findIndex((w) => w.gate === "passed");
  return (
    <>
      <SectionTitle icon={<MathOperations weight="bold" className="size-4" />}>How this segment's score is calculated</SectionTitle>
      <p className="mb-2.5 text-[12.5px] leading-relaxed text-mute">The segment score is the duration-weighted average of <strong className="text-ink">all</strong> its 10-second windows.
        A window with nothing to assess, or that restates nothing, counts as 0. The whole video's score is the same average over every window.</p>
      <div className="overflow-x-auto"><table className="w-full border-collapse text-[13px]"><thead><tr>
        <th className={th}>Window</th><th className={th}>Status</th><th className={cn(th, "text-right")}>Score</th><th className={cn(th, "text-right")}>Duration</th><th className={cn(th, "text-right")}>Score × duration</th></tr></thead>
        <tbody>{ws.map((w) => (
          <tr key={w.index} className={w.qualifies ? "" : "opacity-60"}>
            <td className={td}>{w.timecode}</td>
            <td className={cn(td, "font-sans")}>{w.qualifies ? (w.gate === "passed" ? "Restates a sentence" : "Assessed — nothing restated") : `Nothing to assess — ${w.exclusion_label ?? ""} (counts 0)`}</td>
            <td className={cn(td, "text-right")}>{w.qualifies ? fmt(w.score) : "0"}</td>
            <td className={cn(td, "text-right")}>{f2(w.duration)} s</td>
            <td className={cn(td, "text-right")}>{f2(w.qualifies ? (w.score ?? 0) * w.duration : 0)}</td>
          </tr>))}</tbody></table></div>
      <div className="formula mt-3 rounded-xl bg-s2 px-4 py-3 font-mono text-[13px] leading-7 tabular" data-testid="segment-formula">
        {ws.length ? <>Segment score = ({ws.map((w) => `${w.qualifies ? f2(w.score ?? 0) : "0"} × ${f2(w.duration)}`).join(" + ")}) ÷ ({ws.map((w) => f2(w.duration)).join(" + ")})
          <br />= {f2(num)} ÷ {f2(den)} = <strong className="text-ink">{fmt(seg.score)}</strong></> : "No measurement window falls in this segment."}
      </div>
      <div className="mt-6"><SectionTitle icon={<Stack weight="bold" className="size-4" />}>Window by window <span className="font-normal text-mute">— click a window to open its calculation</span></SectionTitle></div>
      {ws.map((w, i) => <WindowCalc key={w.index} w={w} m={report.methodology} open={i === (firstScored === -1 ? 0 : firstScored)} />)}
    </>
  );
}

function CoherenceSegment({ report, index }: { report: Report; index: number }) {
  const c = report.coherence!;
  const seg = c.segments[index];
  const wts = c.methodology.weights;
  const ws = c.windows.filter((w) => w.start >= seg.start - 1e-6 && w.start < seg.end - 1e-6);
  const num = ws.reduce((a, w) => a + (w.qualifies ? (w.score ?? 0) * w.duration : 0), 0);
  const den = ws.reduce((a, w) => a + w.duration, 0);
  return (
    <>
      <SectionTitle icon={<MathOperations weight="bold" className="size-4" />}>How this segment's coherence is calculated</SectionTitle>
      <p className="mb-2.5 text-[12.5px] leading-relaxed text-mute">Window score = 100 × ({wts.background_music} × share of speech with music under it + {wts.off_topic} × off-topic).
        Off-topic narration is shown but kept out of the score (weight {wts.off_topic}). A window with no narration counts as 0.</p>
      <div className="overflow-x-auto"><table className="w-full border-collapse text-[13px]"><thead><tr>
        <th className={th}>Window</th><th className={cn(th, "text-right")}>Spoken</th><th className={cn(th, "text-right")}>Music under speech</th>
        <th className={cn(th, "text-right")}>Music share</th><th className={th}>Off-topic</th><th className={cn(th, "text-right")}>Score</th></tr></thead>
        <tbody>{ws.map((w) => (
          <tr key={w.index} className={w.qualifies ? "" : "opacity-60"}>
            <td className={td}>{w.timecode}</td>
            <td className={cn(td, "text-right")}>{w.narrated_s.toFixed(1)} s</td>
            <td className={cn(td, "text-right")}>{w.music_under_speech_s.toFixed(1)} s</td>
            <td className={cn(td, "text-right")}>{(w.background_music * 100).toFixed(0)}%</td>
            <td className={cn(td, "font-sans")}>{w.qualifies ? (w.off_topic ? "off-topic" : "on topic") : "no narration (counts 0)"}</td>
            <td className={cn(td, "text-right")}>{w.qualifies ? fmt(w.score) : "0"}</td>
          </tr>))}</tbody></table></div>
      <div className="formula mt-3 rounded-xl bg-s2 px-4 py-3 font-mono text-[13px] leading-7 tabular" data-testid="segment-formula">
        Segment score = {f2(num)} ÷ {f2(den)} = <strong className="text-ink">{fmt(seg.score)}</strong>
      </div>
    </>
  );
}
