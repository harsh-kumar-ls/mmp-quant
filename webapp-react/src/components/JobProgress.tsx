import { Check, CircleNotch } from "@phosphor-icons/react";
import type { Job, Stage } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Banner, Card, CardTitle } from "./ui/primitives";

export function JobProgress({ job, stages }: { job: Job; stages: Stage[] }) {
  if (job.status === "failed") return <Banner tone="bad" title={`Scoring failed — ${job.video_id}`}>{job.error || "unknown error"}</Banner>;
  return (
    <Card>
      <CardTitle sub="Speech recognition and OCR each take one to several minutes, depending on video length.">Scoring {job.video_id}</CardTitle>
      <ol className="space-y-2">
        {stages.filter((s) => s.key !== "done").map((s, i) => {
          const state = job.stage_index > i ? "done" : job.stage_index === i ? "active" : "todo";
          return (
            <li key={s.key} className={cn("flex items-start gap-3 rounded-xl border px-3.5 py-2.5",
              state === "active" ? "border-coral/40 bg-[#f6e4dc]" : "border-line bg-s1", state === "todo" && "opacity-55")}>
              <span className={cn("grid size-8 shrink-0 place-items-center rounded-lg",
                state === "done" ? "bg-[#dcebdf] text-good" : state === "active" ? "bg-coral text-white" : "bg-s3 text-mute")}>
                {state === "done" ? <Check weight="bold" className="size-4" /> : state === "active" ? <CircleNotch weight="bold" className="size-4 animate-spin" /> : <span className="text-xs font-bold">{i + 1}</span>}
              </span>
              <div><div className="text-sm font-semibold">{s.label}</div><div className="text-xs text-mute">{s.description}</div></div>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}
