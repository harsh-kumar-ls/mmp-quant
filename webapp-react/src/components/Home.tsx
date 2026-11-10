import { lazy, Suspense, useMemo, useState } from "react";
import { ArrowRight, Eye, MagnifyingGlass, Microphone, Scales } from "@phosphor-icons/react";
import type { VideoSummary } from "@/lib/api";
import { cn, seqColor } from "@/lib/utils";
import { PenArrow, PenUnderline } from "./ui/marks";

const Scene3D = lazy(() => import("./Scene3D"));

const STYLE_LABEL: Record<string, string> = { slides: "Slides", whiteboard: "Whiteboard", pen: "Pen & paper", animation: "Animation", footage: "Filmed" };

export function Home({ videos, onOpen, onUpload }: { videos: VideoSummary[] | null; onOpen: (id: string) => void; onUpload: (f: File) => void }) {
  return (
    <>
      <Hero count={videos?.length ?? null} />
      <HowItWorks />
      <Library videos={videos} onOpen={onOpen} onUpload={onUpload} />
    </>
  );
}

function Hero({ count }: { count: number | null }) {
  return (
    <section className="grid items-center gap-6 pb-6 pt-6 md:grid-cols-[1.05fr_1fr] md:pt-12">
      <div className="rise">
        <p className="mb-4 inline-flex items-center gap-2 rounded-md bg-s2 px-2.5 py-1 font-mono text-[12px] text-ink-2">
          Mayer's redundancy &amp; coherence principles
        </p>
        <h1 className="text-[44px] leading-[0.98] font-extrabold tracking-[-0.035em] sm:text-[64px]">
          Is the screen just{" "}
          <span className="relative inline-block whitespace-nowrap">reading
            <PenUnderline className="absolute -bottom-2 left-0 h-4 w-full" /></span>{" "}
          the narration back to you?
        </h1>
        <p className="mt-6 max-w-[54ch] text-[17px] leading-relaxed text-ink-2">
          MMP-Quant watches an educational video the way a careful marker would: it reads every sentence on screen,
          listens to what is said at that moment, and marks the places where the two say the <span className="hl">same thing twice</span>.
          Every mark comes with its working.
        </p>
        <p className="mt-5 font-mono text-[13px] text-mute">
          {count === null ? "Loading the library…" : `${count} videos marked so far · scores 0–100 · higher means more redundant`}
        </p>
      </div>
      <div className="relative h-[340px] sm:h-[420px]">
        <Suspense fallback={<div className="size-full animate-pulse rounded-[28px] bg-s2/60" />}>
          <Scene3D />
        </Suspense>
        <p className="pointer-events-none absolute bottom-1 right-2 font-mono text-[11px] text-mute">drag to turn it</p>
      </div>
    </section>
  );
}

function HowItWorks() {
  const steps = [
    { icon: Eye, title: "It reads the screen", body: "OCR finds the on-screen text. Only whole sentences can be redundant — titles, labels and highlighted key lines are signposts, so they are set aside." },
    { icon: Microphone, title: "It listens", body: "Speech recognition writes down the narration with a time for every word, so a sentence is only compared with what is said while it is up." },
    { icon: Scales, title: "It marks, and shows the working", body: "A sentence counts when the narration restates it — more than half its words, above chance for that video. Click any segment to check the sum yourself." },
  ];
  return (
    <section className="relative my-10 grid gap-x-10 gap-y-8 rounded-[28px] bg-s2/60 px-6 py-9 sm:px-10 md:grid-cols-[0.8fr_1.2fr]">
      <div>
        <h2 className="text-[30px] leading-tight font-extrabold tracking-tight">How the marking works</h2>
        <p className="mt-3 max-w-[38ch] text-[15px] leading-relaxed text-ink-2">
          Three passes, in the order a teacher would do them. Nothing is a black box — the numbers on every page add up.
        </p>
        <PenArrow className="mt-4 hidden h-16 w-28 md:block" />
      </div>
      <ol className="space-y-6">
        {steps.map(({ icon: Icon, title, body }, i) => (
          <li key={title} className={cn("rise flex gap-4", i === 1 && "md:ml-10", i === 2 && "md:ml-4")} style={{ animationDelay: `${120 + i * 90}ms` }}>
            <span className="display grid size-11 shrink-0 place-items-center rounded-xl bg-s1 text-[18px] font-extrabold text-coral shadow-[var(--shadow-paper)]">{i + 1}</span>
            <div>
              <h3 className="flex items-center gap-2 text-[17px] font-bold"><Icon weight="duotone" className="size-5 text-ink-2" />{title}</h3>
              <p className="mt-1 max-w-[56ch] text-[14.5px] leading-relaxed text-ink-2">{body}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Library({ videos, onOpen, onUpload }: { videos: VideoSummary[] | null; onOpen: (id: string) => void; onUpload: (f: File) => void }) {
  const [q, setQ] = useState("");
  const [style, setStyle] = useState<string>("all");
  const [sort, setSort] = useState<"score" | "name">("score");
  const styles = useMemo(() => [...new Set((videos ?? []).map((v) => v.style).filter(Boolean))] as string[], [videos]);
  const shown = useMemo(() => (videos ?? [])
    .filter((v) => (style === "all" || v.style === style) && v.video_id.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => sort === "name" ? a.video_id.localeCompare(b.video_id) : (b.score ?? -1) - (a.score ?? -1)), [videos, q, style, sort]);

  return (
    <section id="library" className="pb-10">
      <div className="mb-5 flex flex-wrap items-end gap-x-6 gap-y-3">
        <div>
          <h2 className="text-[30px] leading-tight font-extrabold tracking-tight">The marked pile</h2>
          <p className="mt-1 text-[14px] text-mute">Every video scored so far. Open one to see its marks.</p>
        </div>
        <label className="relative ml-auto w-full sm:w-64">
          <span className="sr-only">Search videos</span>
          <MagnifyingGlass className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-mute" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by name"
            className="w-full rounded-xl border border-line bg-s1 py-2 pl-9 pr-3 text-[14px] placeholder:text-mute focus:border-coral focus:outline-none" />
        </label>
      </div>
      <div className="mb-4 flex flex-wrap items-center gap-2" role="group" aria-label="Filter by style">
        {["all", ...styles].map((s) => (
          <button key={s} type="button" onClick={() => setStyle(s)} aria-pressed={style === s}
            className={cn("rounded-lg px-3 py-1.5 text-[13px] font-medium transition active:scale-[.98]",
              style === s ? "bg-ink text-bg" : "bg-s2 text-ink-2 hover:bg-s3")}>
            {s === "all" ? "All styles" : STYLE_LABEL[s] ?? s}
          </button>
        ))}
        <button type="button" onClick={() => setSort(sort === "score" ? "name" : "score")}
          className="ml-auto text-[13px] text-ink-2 underline decoration-line-strong underline-offset-4 hover:text-ink">
          Sorted by {sort === "score" ? "score" : "name"}
        </button>
      </div>

      <div className="paper overflow-hidden rounded-[20px]" data-testid="video-list">
        {videos === null && <div className="p-6 text-sm text-mute">Loading the library…</div>}
        {videos !== null && shown.length === 0 && (
          <div className="p-8 text-center">
            <p className="text-[15px] text-ink-2">{videos.length === 0 ? "Nothing marked yet." : "No video matches that search."}</p>
            {videos.length === 0 && (
              <label className="mt-3 inline-block cursor-pointer rounded-xl bg-ink px-4 py-2 text-sm font-semibold text-bg">
                Mark your first video
                <input type="file" hidden accept=".mp4,.mov,.mkv,.webm,.m4v,.avi" onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); }} />
              </label>)}
          </div>
        )}
        <ul className="divide-y divide-line">
          {shown.map((v, i) => (
            <li key={v.video_id}>
              <button type="button" onClick={() => onOpen(v.video_id)}
                className="group grid w-full grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1.5 px-4 py-3 text-left transition hover:bg-[#f6efe3] sm:grid-cols-[minmax(0,1.3fr)_110px_minmax(120px,1fr)_70px_24px] sm:px-6">
                <span className="min-w-0 truncate font-medium">{v.video_id}</span>
                <span className="hidden text-[13px] text-mute sm:block">{v.style ? STYLE_LABEL[v.style] ?? v.style : "—"}</span>
                <span className="col-span-2 h-2 overflow-hidden rounded-full bg-s3 sm:col-span-1" aria-hidden>
                  {v.score !== null && <span className="bar-grow block h-full rounded-full" style={{ width: `${Math.max(2, v.score)}%`, background: seqColor(v.score), animationDelay: `${Math.min(i, 12) * 30}ms` }} />}
                </span>
                <span className="row-start-1 text-right font-mono text-[15px] font-semibold tabular sm:row-start-auto">{v.score === null ? "N/A" : v.score.toFixed(1)}</span>
                <ArrowRight className="hidden size-4 text-mute transition group-hover:translate-x-0.5 group-hover:text-coral sm:block" />
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
