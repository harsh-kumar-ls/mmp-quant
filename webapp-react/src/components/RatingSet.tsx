import { useEffect, useState } from "react";
import { FileText, Target } from "@phosphor-icons/react";
import { api, type RatingSet as RS } from "@/lib/api";
import { fmt, seqColor, timecode } from "@/lib/utils";
import { Banner, Card, CardTitle, Spinner, Stat } from "./ui/primitives";

/* The 10 videos chosen for the professor's ratings (10 Nov 2026).
   STUDENT ONLY: it shows system scores; a rater who saw them would anchor. */
export function RatingSet({ onOpen }: { onOpen: (id: string) => void }) {
  const [data, setData] = useState<RS | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<RS>("/api/rating-set").then(setData).catch((e) => setError(e.message)); }, []);

  if (error) return <Banner tone="bad" title="Could not load the rating set">{error}</Banner>;
  if (!data) return <Spinner label="Loading the rating set…" />;

  const rows = data.rows;
  const score = (r: RS["rows"][number]) => (r.current ? r.current.redundancy : r.redundancy);
  const styles = [...new Set(rows.map((r) => r.style))];
  const music = rows.filter((r) => (r.coherence ?? 0) > 10).length;
  const minutes = rows.reduce((a, r) => a + (r.duration_s ?? 0), 0) / 60;
  const drift = rows.filter((r) => r.current && r.redundancy !== null && score(r) !== null
    && Math.abs((score(r) as number) - r.redundancy) > 0.01);

  return (
    <div className="space-y-4">
      <header>
        <div className="font-mono text-[12px] text-mute">{data.package}</div>
        <h1 className="mt-1 text-[34px] leading-tight font-extrabold tracking-[-0.03em] sm:text-[44px]">The professor's pile</h1>
        <p className="mt-1 text-[13px] text-mute">The 10 videos the professor rates, one per redundancy target 0, 10, … 90,
          chosen by the pre-registered rule. Click a video to open its full report.</p>
      </header>

      <Banner title="Student only — never show this page to the professor.">
        It holds the system's scores. The professor rates blind; seeing these numbers first would anchor the ratings and
        spoil the validation. Send only the package folder's videos, workbook and guide — never <code>selection.csv</code>.
      </Banner>

      <dl className="paper flex flex-wrap gap-x-10 gap-y-4 rounded-[20px] px-6 py-5">
        <Stat label="Videos">{rows.length}</Stat>
        <Stat label="Total distance">{data.total_distance.toFixed(1)}</Stat>
        <Stat label="Styles">{styles.length} <span className="font-sans text-[13px] font-normal text-ink-2">· {styles.join(", ")}</span></Stat>
        <Stat label="With music">{music} of {rows.length}</Stat>
        <Stat label="Watching time">{minutes.toFixed(0)} min</Stat>
      </dl>

      {drift.length > 0 && (
        <Banner title="Scores changed since selection">
          {drift.map((r) => `${r.video_id}: ${fmt(r.redundancy)} → ${fmt(score(r))}`).join("; ")}. The bars show today's score.
        </Banner>
      )}

      <Card>
        <CardTitle icon={<Target weight="bold" className="size-5" />}
          sub="Blue bar = the system's score; red tick = the target it was picked for; the small number = how far off it lands.">
          Score against target
        </CardTitle>
        <div className="flex flex-col gap-1.5" data-testid="rs-ladder">
          {rows.map((r, i) => {
            const s = score(r);
            const d = s === null ? null : s - r.target;
            return (
              <button key={r.video_id} type="button" onClick={() => onOpen(r.video_id)} data-testid="rs-row"
                className="grid grid-cols-[36px_minmax(0,220px)_1fr_74px] items-center gap-3 rounded-xl px-2 py-1.5 text-left transition hover:bg-[#f6efe3] active:scale-[.995]">
                <span className="rounded-md bg-ink py-0.5 text-center font-mono text-xs font-bold text-bg tabular" data-testid="rs-target">{r.target}</span>
                <span className="min-w-0">
                  <span className="block truncate text-[13px]">{r.video_id}</span>
                  <span className="block text-[11px] text-mute">{r.style}</span>
                </span>
                <span className="relative h-[22px] rounded-md bg-s3">
                  {s !== null && <span className="bar-grow block h-full rounded-md" style={{ width: `${Math.max(1, s)}%`, background: seqColor(s), animationDelay: `${i * 50}ms` }} />}
                  <span className="absolute -top-1.5 -bottom-1.5 w-[3px] -ml-[1.5px] rounded-full bg-coral shadow-[0_0_0_2px_var(--color-s1)]" style={{ left: `${r.target}%` }} title={`target ${r.target}`} />
                </span>
                <span className="text-right font-mono text-[14px] font-semibold tabular">{fmt(s)}
                  <span className="block text-[11px] font-normal text-mute">{d === null ? "" : `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`}</span>
                </span>
              </button>
            );
          })}
        </div>
        <div className="grid grid-cols-[36px_minmax(0,220px)_1fr_74px] gap-3 px-2 pt-2 font-mono text-[11px] text-mute">
          <span className="col-span-2" /><span className="flex justify-between"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></span><span />
        </div>
      </Card>

      <Card>
        <CardTitle icon={<FileText weight="bold" className="size-5" />}>Details</CardTitle>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead><tr className="text-[10.5px] uppercase tracking-wide text-mute">
              {["Target", "Video", "Style", "Length", "Redundancy", "Distance", "Coherence"].map((h, i) => (
                <th key={h} className={`border-b border-line px-2 py-2 font-semibold ${i === 1 || i === 2 ? "text-left" : "text-right"}`}>{h}</th>))}
            </tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.video_id} onClick={() => onOpen(r.video_id)} className="cursor-pointer text-ink-2 hover:bg-s2 hover:text-ink">
                <td className="border-b border-line px-2 py-2 text-right tabular">{r.target}</td>
                <td className="border-b border-line px-2 py-2">{r.video_id}</td>
                <td className="border-b border-line px-2 py-2">{r.style}</td>
                <td className="border-b border-line px-2 py-2 text-right tabular">{timecode(r.duration_s ?? 0)}</td>
                <td className="border-b border-line px-2 py-2 text-right tabular">{fmt(score(r))}</td>
                <td className="border-b border-line px-2 py-2 text-right tabular">{fmt(r.distance)}</td>
                <td className="border-b border-line px-2 py-2 text-right tabular">{fmt(r.current ? r.current.coherence : r.coherence)}
                  {(r.coherence ?? 0) > 10 && <span className="text-mute"> (music)</span>}</td>
              </tr>))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
