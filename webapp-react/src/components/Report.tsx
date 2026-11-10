import { useState } from "react";
import { ArrowRight, Eye, Microphone, MusicNotes, NotePencil, SlidersHorizontal, Warning } from "@phosphor-icons/react";
import type { Report as R } from "@/lib/api";
import { cn, fmt, pct, seqColor, timecode } from "@/lib/utils";
import { Badge, Banner, Card, CardTitle, ScoreRing, Stat } from "./ui/primitives";
import { SegmentDrawer, type Principle } from "./SegmentDrawer";

export function Report({ data }: { data: R }) {
  const [tab, setTab] = useState<Principle>("redundancy");
  const [open, setOpen] = useState<{ p: Principle; i: number } | null>(null);
  const coh = data.coherence;

  return (
    <div className="space-y-6">
      <header className="rise">
        <p className="font-mono text-[12px] text-mute">{timecode(data.duration)} · {data.n_windows_total} windows of 10 s</p>
        <h1 className="mt-1 break-all text-[34px] leading-tight font-extrabold tracking-[-0.03em] sm:text-[44px]">{data.video_id}</h1>
        <div className="mt-5 flex gap-6 border-b border-line" role="tablist">
          <TabBtn active={tab === "redundancy"} onClick={() => setTab("redundancy")} testid="tab-redundancy"
            title="Redundancy" score={data.score} />
          <TabBtn active={tab === "coherence"} onClick={() => setTab("coherence")} testid="tab-coherence"
            title="Coherence" score={coh?.score ?? null} />
        </div>
      </header>

      {tab === "redundancy" ? (
        <>
          <Card>
            <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
              <ScoreRing score={data.score} />
              <div className="min-w-0 flex-1">
                <p className="max-w-[60ch] text-[15px] leading-relaxed text-ink-2">
                  <strong className="text-ink">Redundancy score.</strong> Higher means more on-screen sentences are read out while they are on screen —
                  worse, per Mayer. It is the average over every 10-second window; a window with nothing to compare counts as 0.
                </p>
                <dl className="mt-4 flex flex-wrap gap-x-8 gap-y-3">
                  <Stat label="Windows assessed">{data.n_windows_qualifying} / {data.n_windows_total}</Stat>
                  <Stat label="Where assessed">{fmt(data.assessed_score)}</Stat>
                  <Stat label="Coverage">{pct(data.coverage)}</Stat>
                  <Stat label="Status"><Badge tone={data.status === "OK" ? "ok" : "warn"}>{data.status === "OK" ? "Marked" : data.status}</Badge></Stat>
                </dl>
              </div>
            </div>
            {data.na_reason && <div className="mt-5"><Banner title="Not marked">{data.na_reason}</Banner></div>}
            {data.warnings.length > 0 && (
              <details className="mt-5 rounded-xl bg-[#f6ebd3] px-4 py-3 text-[14px] text-[#4d3500]">
                <summary className="flex cursor-pointer items-center gap-2 font-medium"><Warning weight="bold" className="size-4" />{data.warnings.length} note{data.warnings.length > 1 ? "s" : ""} about this run</summary>
                <ul className="mt-2 list-disc space-y-1 pl-5 leading-relaxed">{data.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
              </details>)}
          </Card>
          <Timeline segs={data.segments.map((s) => ({ index: s.index, timecode: s.timecode, score: s.score }))}
            title="Segment by segment" onOpen={(i) => setOpen({ p: "redundancy", i })}
            sub={<>Thirty-second segments are what the professor rates. <strong className="text-ink">Click one</strong> to see its text, the narration with word times, and every step of the sum.</>} />
          <Worst data={data} onOpen={(i) => setOpen({ p: "redundancy", i })} />
          <Method data={data} />
        </>
      ) : coh && coh.score !== null ? (
        <>
          <Card>
            <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
              <ScoreRing score={coh.score} />
              <div className="min-w-0 flex-1">
                <p className="max-w-[60ch] text-[15px] leading-relaxed text-ink-2">
                  <strong className="text-ink">Coherence score.</strong> Higher means more extraneous material — mostly background music
                  playing under the narration. Off-topic talk is detected and shown, but kept out of the score.
                </p>
                <dl className="mt-4 flex flex-wrap gap-x-8 gap-y-3">
                  <Stat label="Music under speech">{fmt(coh.parts.background_music)}</Stat>
                  <Stat label="Off-topic (not scored)">{fmt(coh.parts.off_topic)}</Stat>
                  <Stat label="Music loudness">{coh.music_loudness.median_db === null ? "—" : `${coh.music_loudness.median_db.toFixed(1)} dB`}</Stat>
                </dl>
              </div>
            </div>
          </Card>
          <Timeline segs={coh.segments.map((s) => ({ index: s.index, timecode: s.timecode, score: s.score }))}
            title="Music, segment by segment" principle="coherence" onOpen={(i) => setOpen({ p: "coherence", i })}
            sub="Share of the narration with background music under it. Click a segment for its windows." />
        </>
      ) : (
        <Banner title="Coherence not available">{coh?.na_reason ?? "Not scored in this run — re-score the video."}</Banner>
      )}

      {open && <SegmentDrawer report={data} principle={open.p} index={open.i} onIndex={(i) => setOpen({ ...open, i })} onClose={() => setOpen(null)} />}
    </div>
  );
}

function TabBtn({ active, onClick, title, score, testid }: { active: boolean; onClick: () => void; title: string; score: number | null; testid: string }) {
  return (
    <button type="button" role="tab" aria-selected={active} onClick={onClick} data-testid={testid}
      className={cn("relative -mb-px flex items-center gap-2 pb-3 pt-1 text-[15px] font-semibold transition",
        active ? "text-ink after:absolute after:inset-x-0 after:-bottom-px after:h-[3px] after:rounded-full after:bg-coral" : "text-mute hover:text-ink")}>
      {title === "Coherence" ? <MusicNotes weight="bold" className="size-4" /> : <NotePencil weight="bold" className="size-4" />}
      {title}<span className="font-mono text-[13px] font-medium text-ink-2 tabular">{fmt(score, 1)}</span>
    </button>
  );
}

function Timeline({ segs, title, sub, onOpen, principle = "redundancy" }: {
  segs: { index: number; timecode: string; score: number | null }[]; title: string; sub: React.ReactNode; onOpen: (i: number) => void; principle?: Principle;
}) {
  return (
    <Card>
      <CardTitle sub={sub}>{title}</CardTitle>
      <div className="flex flex-col gap-1">
        {segs.map((s, i) => (
          <button key={s.index} type="button" onClick={() => onOpen(i)} data-testid="segment-row" data-principle-of={principle}
            aria-label={`Show how segment ${i + 1} (${s.timecode}) was scored`}
            className="group grid grid-cols-[84px_1fr_64px_20px] items-center gap-3 rounded-xl px-2 py-1.5 text-left transition hover:bg-[#f6efe3] active:scale-[.995]">
            <span className="font-mono text-[13px] text-ink-2 tabular">{s.timecode}</span>
            <span className="h-[22px] overflow-hidden rounded-md bg-s3">
              <span className="bar-grow block h-full min-w-[3px] rounded-md" style={{ width: `${Math.max(1, s.score ?? 0)}%`, background: seqColor(s.score), animationDelay: `${i * 60}ms` }} />
            </span>
            <span className="text-right font-mono text-[14px] font-semibold tabular" data-testid="segment-value">{fmt(s.score)}</span>
            <ArrowRight className="size-4 text-mute transition group-hover:translate-x-0.5 group-hover:text-coral" />
          </button>
        ))}
      </div>
      <div className="grid grid-cols-[84px_1fr_64px_20px] gap-3 px-2 pt-2 font-mono text-[11px] text-mute">
        <span /><span className="flex justify-between"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></span><span /><span />
      </div>
    </Card>
  );
}

function Worst({ data, onOpen }: { data: R; onOpen: (i: number) => void }) {
  if (!data.worst_segments.length) return null;
  return (
    <Card>
      <CardTitle sub={<>The evidence behind the highest marks. <span className="hl">Highlighted</span> = the sentences the narration was found to restate. Read the two columns side by side and judge for yourself.</>}>
        Where it repeats itself most
      </CardTitle>
      <div className="space-y-7">
        {data.worst_segments.map((s) => (
          <article key={s.segment_index}>
            <button type="button" onClick={() => onOpen(s.segment_index)}
              className="inline-flex items-center gap-2 font-mono text-[13px] font-semibold text-ink hover:text-coral">
              Segment {s.segment_index + 1} · {s.timecode} · <span className="text-coral">{fmt(s.score)}</span><ArrowRight className="size-3.5" />
            </button>
            {s.windows.slice(0, 2).map((w, i) => (
              <div key={i} className="mt-3 grid gap-3 md:grid-cols-2">
                <div className="rounded-xl bg-s2/70 p-4 text-[14px] leading-relaxed">
                  <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-semibold text-teal"><Eye weight="bold" className="size-3.5" />On screen · {w.timecode}</div>
                  <Marked text={w.onscreen_text} restated={w.restated_sentences} />
                </div>
                <div className="rounded-xl bg-s2/70 p-4 text-[14px] leading-relaxed text-ink-2">
                  <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-semibold text-rose"><Microphone weight="bold" className="size-3.5" />Narration · window {fmt(w.score)}</div>
                  {w.narration_text}
                </div>
              </div>))}
          </article>))}
      </div>
    </Card>
  );
}

/** The on-screen text with ONLY the restated sentences highlighted — the
    highlighter must not claim more was read aloud than the system found. */
function Marked({ text, restated }: { text: string; restated: string[] }) {
  const flat = text.replace(/\s+/g, " ");
  const hits: [number, number][] = [];
  for (const r of restated) {
    const needle = r.replace(/\s+/g, " ").trim();
    let from = 0, at: number;
    while (needle && (at = flat.indexOf(needle, from)) !== -1) { hits.push([at, at + needle.length]); from = at + needle.length; }
  }
  hits.sort((x, y) => x[0] - y[0]);
  const out: React.ReactNode[] = [];
  let pos = 0;
  hits.forEach(([a, b], i) => {
    if (a < pos) return;
    if (a > pos) out.push(<span key={`t${i}`} className="text-ink-2">{flat.slice(pos, a)}</span>);
    out.push(<span key={`h${i}`} className="hl">{flat.slice(a, b)}</span>);
    pos = b;
  });
  if (pos < flat.length) out.push(<span key="end" className="text-ink-2">{flat.slice(pos)}</span>);
  return <>{out}</>;
}

function Method({ data }: { data: R }) {
  const m = data.methodology;
  const rows: [string, React.ReactNode][] = [
    ["Window", `${m.window_size_s} s`], ["Segment (rated unit)", `${m.segment_length_s} s`],
    ["Sentence score", `${m.weight_verbatim} × word match + ${m.weight_semantic} × meaning`],
    ["Restated", `> ${pct(m.min_sentence_coverage)} of content words, ≥ ${m.min_shared_content_words}, above chance`],
    ["Timing", `while on screen, up to ${m.nearby_reach_s ?? 10} s past the window, ± ${m.alignment_tolerance_s} s`],
    ["Chance baseline", `narration > ${m.null_exclusion_s} s away`], ["Signaling", `sentence text ≥ ${m.min_onscreen_words} words`],
    ["Embeddings", m.embedding_model], ["ASR / OCR", `${m.asr_backend} / ${m.ocr_backend}, ${m.frame_fps} fps`],
  ];
  return (
    <details className="group rounded-[20px] bg-s2/60 px-5 py-4 sm:px-7">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-[16px] font-bold [&::-webkit-details-marker]:hidden">
        <SlidersHorizontal weight="bold" className="size-4 text-coral" />The settings behind these marks
        <span className="ml-auto font-mono text-[12px] font-normal text-mute group-open:hidden">show</span>
      </summary>
      <dl className="mt-4 grid gap-x-8 sm:grid-cols-2">
        {rows.map(([k, v]) => <div key={k} className="flex justify-between gap-4 border-b border-line py-2 text-[14px]"><dt className="text-mute">{k}</dt><dd className="m-0 text-right font-mono text-[13px] text-ink-2">{v}</dd></div>)}
      </dl>
    </details>
  );
}
