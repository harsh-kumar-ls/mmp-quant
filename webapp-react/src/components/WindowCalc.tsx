import { CaretRight, Check, Eye, Info, Microphone } from "@phosphor-icons/react";
import type { Methodology, SentenceCheck, WindowBreakdown } from "@/lib/api";
import { cn, fmt, pct, tc } from "@/lib/utils";
import { Badge } from "./ui/primitives";

const f4 = (x?: number | null) => (x === null || x === undefined ? "—" : Number(x).toFixed(4));
const spansText = (spans?: [number, number][]) => (spans || []).map(([a, b]) => `${tc(a)}–${tc(b)}`).join(", ");

function checkResult(c: SentenceCheck, m: Methodology) {
  const need = m.min_sentence_coverage ?? 0.5, minShared = m.min_shared_content_words ?? 2;
  if (c.passed) return <span className="inline-flex items-center gap-1 font-semibold text-good"><Check weight="bold" className="size-3.5" />Restated</span>;
  const fail = (t: string) => <span className="font-semibold text-bad">✗ {t}</span>;
  if (c.is_question && !c.shared_words.length) return fail("Not asked aloud as a question");
  if (c.shared_words.length < minShared) return fail(`Fewer than ${minShared} words said`);
  if (c.coverage <= need) return fail(`Not more than ${pct(need)}`);
  return fail("Not above chance");
}

function chanceCell(raw?: number, chance?: number, adj?: number) {
  if (raw === null || raw === undefined) return "—";
  return raw <= (chance ?? 0)
    ? <>{f4(raw)} ≤ chance {f4(chance)} → <strong>0</strong></>
    : <>({f4(raw)} − {f4(chance)}) ÷ (1 − {f4(chance)}) = <strong>{f4(adj)}</strong></>;
}

/** Narration word by word with a time stamp at each pause; words spoken in the
    ± tolerance margin outside the window are faded (user, 21 Oct 2026). */
function TimedNarration({ w }: { w: WindowBreakdown }) {
  const words = w.narration_timed || [];
  if (!words.length) return <>{w.narration || <em>(no narration)</em>}</>;
  const out: React.ReactNode[] = [];
  let last: number | null = null;
  words.forEach((x, i) => {
    if (last === null || x.start - last > 1.0) out.push(<Stamp key={`t${i}`}>{tc(x.start)}</Stamp>);
    out.push(<span key={i} title={`${tc(x.start)}–${tc(x.end)}`} className={cn(x.in_window ? "w-in" : "w-outside text-mute opacity-60")}>{x.text} </span>);
    last = x.end;
  });
  return <>{out}</>;
}

const Stamp = ({ children }: { children: React.ReactNode }) => (
  <span className="tstamp mr-1 inline-block rounded-md bg-s3 px-1.5 font-mono text-[11px] text-ink-2 tabular">{children}</span>
);

const th = "border-b border-line px-2 py-2 text-left text-[10.5px] font-semibold uppercase tracking-wide text-mute";
const td = "border-b border-line px-2 py-2 align-top text-ink-2";

export function WindowCalc({ w, m, open }: { w: WindowBreakdown; m: Methodology; open: boolean }) {
  const status = !w.qualifies ? ["warn", "Left out"] : w.gate === "passed" ? ["scored", "Restates a sentence"] : ["neutral", "Nothing restated"];
  const tol = String(m.alignment_tolerance_s ?? 3);
  const checks = w.sentence_checks || [];
  const timed = checks.some((c) => c.spans);

  const texts = (
    <>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <div className="rounded-xl bg-s2/70 p-3.5 text-[13.5px] leading-relaxed text-ink">
          <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-semibold text-teal"><Eye weight="bold" className="size-3.5" />On-screen sentences{timed ? " — when shown" : ""}</div>
          {timed ? checks.map((c, i) => <div key={i} className="mt-1.5 first:mt-0"><Stamp>{spansText(c.spans)}</Stamp><span className={c.passed ? "hl" : ""}>{c.sentence}</span></div>)
            : (w.onscreen_compared || <em>(no sentence on screen)</em>)}
        </div>
        <div className="rounded-xl bg-s2/70 p-3.5 text-[13.5px] leading-relaxed text-ink" data-testid="voice">
          <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-semibold text-rose"><Microphone weight="bold" className="size-3.5" />Narration (± {tol} s)</div>
          <TimedNarration w={w} />
        </div>
      </div>
      {w.narration_timed?.length > 0 && <p className="mt-2 text-xs text-mute">Times are when each word is spoken (hover a word). Faded words fall in the ± {tol} s margin outside {w.timecode}.</p>}
      {w.labels_excluded.length > 0 && (
        <div className="mt-2.5">
          <p className="text-xs text-mute">Set aside as labels / key terms — signaling, never redundancy:</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">{w.labels_excluded.map((l, i) => <span key={i} className="rounded-md bg-s3 px-2 py-0.5 text-xs text-ink-2">{l}</span>)}
            {w.n_labels_excluded > w.labels_excluded.length && <span className="rounded-md bg-s3 px-2 py-0.5 text-xs text-mute">+{w.n_labels_excluded - w.labels_excluded.length} more</span>}</div>
        </div>)}
    </>
  );

  let body: React.ReactNode;
  if (!w.qualifies) {
    body = <>{texts}<Verdict><strong>Not assessed.</strong> {w.exclusion_label ? w.exclusion_label[0].toUpperCase() + w.exclusion_label.slice(1) : ""}.
      A window needs at least {m.min_onscreen_words} words of on-screen sentence text and some narration, so there is nothing here to assess — it counts as 0 in the segment average.</Verdict></>;
  } else {
    const step1 = (
      <Step n={1} title="Does the narration restate an on-screen sentence?">
        <p className="mb-2.5 text-[12.5px] leading-relaxed text-mute">Each sentence is compared with the narration spoken while it is on screen — up to {m.nearby_reach_s ?? 10} s beyond this window either side — ± {tol} s.
          Restated = that narration says at least {m.min_shared_content_words ?? 2} of the sentence's content words, <strong className="text-ink">more than {pct(m.min_sentence_coverage ?? 0.5)}</strong> of them,
          and more than narration elsewhere in the video does (chance: median over {w.null_windows || 0} windows &gt; {m.null_exclusion_s ?? 30} s away{w.null_windows ? "" : " — too few, so taken as 0"}).
          A question counts only if it is asked aloud.</p>
        {checks.length ? (
          <div className="overflow-x-auto"><table className="w-full border-collapse text-[13px]"><thead><tr>
            <th className={th}>On-screen sentence</th><th className={th}>Its content words</th><th className={th}>Narration said</th>
            <th className={cn(th, "text-right")}>Coverage</th><th className={cn(th, "text-right")}>Chance</th><th className={th}>Result</th></tr></thead>
            <tbody>{checks.map((c, i) => (
              <tr key={i} className={c.passed ? "bg-[#e6efe2]" : ""}>
                <td className={td}>{c.sentence}{c.is_question && <span className="text-mute"> (question)</span>}
                  {c.narration_from !== null && c.narration_from !== undefined && (
                    <div className="mt-1 text-xs text-mute"><Stamp>{tc(c.narration_from)}–{tc(c.narration_to!)}</Stamp>compared with: {c.narration || "(nothing spoken)"}</div>)}</td>
                <td className={td}>{c.content_words.join(", ")}</td><td className={td}>{c.shared_words.join(", ") || "—"}</td>
                <td className={cn(td, "text-right tabular")}>{c.shared_words.length}/{c.content_words.length} = {pct(c.coverage)}</td>
                <td className={cn(td, "text-right tabular")}>{pct(c.chance)}</td><td className={td}>{checkResult(c, m)}</td>
              </tr>))}</tbody></table></div>
        ) : <p className="text-xs italic text-mute">No full sentence on screen in this window to test.</p>}
      </Step>
    );
    if (w.gate !== "passed") {
      body = <>{texts}{step1}<Verdict>No on-screen sentence is restated → <strong>window score = 0</strong> (assessed, no redundancy). It still counts as 0 in the segment average.</Verdict></>;
    } else if (checks.some((c) => c.time_weight === undefined)) {
      body = <>{texts}{step1}<Verdict>Window score <strong>{fmt(w.score)}</strong>. This run was scored before the time-weighted calculation (21 Oct 2026); re-score the video to see every step.</Verdict></>;
    } else {
      const wv = m.weight_verbatim, ws = m.weight_semantic;
      const shown = (w.screen_time_share ?? 1) * w.duration;
      const counted = checks.filter((c) => (c.time_weight ?? 0) > 0);
      body = <>{texts}{step1}
        <Step n={2} title="Score each sentence">
          <p className="mb-2.5 text-[12.5px] leading-relaxed text-mute">Word match = share of the sentence's content words the narration says, in order. Meaning = cosine similarity with the best-matching narration sentence ({m.embedding_model}).
            Only what is above chance counts: (raw − chance) ÷ (1 − chance). Sentence score = 100 × ({wv} × word match + {ws} × meaning); a sentence not restated scores 0.</p>
          <div className="overflow-x-auto"><table className="w-full border-collapse text-[13px]"><thead><tr>
            <th className={th}>Sentence</th><th className={th}>On screen</th><th className={th}>Word match</th><th className={th}>Meaning</th><th className={cn(th, "text-right")}>Sentence score</th></tr></thead>
            <tbody>{checks.map((c, i) => (
              <tr key={i} className={c.passed ? "bg-[#e6efe2]" : ""}>
                <td className={td}>{c.sentence}</td>
                <td className={cn(td, "tabular")}>{spansText(c.spans)}<div className="text-xs text-mute">{(c.visible_s ?? 0).toFixed(1)} s in this window</div></td>
                <td className={cn(td, "tabular")}>{c.passed ? chanceCell(c.verbatim_raw, c.verbatim_chance, c.verbatim) : "not restated"}</td>
                <td className={cn(td, "tabular")}>{c.passed ? chanceCell(c.semantic_raw, c.semantic_chance, c.semantic) : "—"}</td>
                <td className={cn(td, "text-right tabular")}><strong className="text-ink">{fmt(c.score ?? 0)}</strong></td>
              </tr>))}</tbody></table></div>
        </Step>
        <Step n={3} title="Window score — weighted by time on screen">
          <p className="mb-2.5 text-[12.5px] leading-relaxed text-mute">Every half-second of the window with a sentence on screen counts equally ({shown.toFixed(1)} of {w.duration.toFixed(0)} s here; moments with no sentence are left out).
            Two sentences on screen together share the moment in proportion to their content words. Each sentence's weight is its share of those moments.</p>
          <div className="formula rounded-xl bg-s2 px-4 py-3 font-mono text-[13px] leading-7 tabular" data-testid="formula">
            {counted.map((c, i) => <div key={i}>{i ? "+ " : ""}{f4(c.time_weight)} × {fmt(c.score ?? 0)}</div>)}
            <div>= <strong className="text-ink">{fmt(w.score)}</strong></div>
          </div>
        </Step></>;
    }
  }

  return (
    <details open={open} className="win group mt-2.5 rounded-2xl bg-s1 shadow-[var(--shadow-paper)]" data-testid="win">
      <summary className="flex cursor-pointer list-none items-center gap-3 px-3.5 py-2.5 [&::-webkit-details-marker]:hidden">
        <CaretRight weight="bold" className="size-4 text-mute transition group-open:rotate-90" />
        <span className="win-time font-mono text-sm font-semibold tabular">{w.timecode}</span>
        <Badge tone={status[0] as "warn"}>{status[1]}</Badge>
        <span className="ml-auto font-mono text-sm font-semibold tabular">{w.qualifies ? fmt(w.score) : "—"}</span>
      </summary>
      <div className="border-t border-line px-3.5 pb-4">{body}</div>
    </details>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="mt-4">
      <div className="mb-2 flex items-center gap-2.5 text-[15px] font-bold"><span className="display grid size-7 place-items-center rounded-lg bg-ink text-[13px] font-extrabold text-bg">{n}</span>{title}</div>
      {children}
    </div>
  );
}

function Verdict({ children }: { children: React.ReactNode }) {
  return <div className="mt-4 flex gap-2.5 rounded-xl bg-s2 px-4 py-3 text-[13.5px] leading-relaxed text-ink-2"><Info weight="bold" className="mt-0.5 size-4 shrink-0 text-coral" /><div>{children}</div></div>;
}
