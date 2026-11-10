import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft } from "@phosphor-icons/react";
import { api, type Job, type Report as R, type Stage, type VideoSummary } from "@/lib/api";
import { TopBar } from "@/components/TopBar";
import { Home } from "@/components/Home";
import { RatingSet } from "@/components/RatingSet";
import { Report } from "@/components/Report";
import { JobProgress } from "@/components/JobProgress";
import { Banner, Spinner } from "@/components/ui/primitives";

type View =
  | { kind: "home" }
  | { kind: "rating" }
  | { kind: "loading"; id: string }
  | { kind: "report"; data: R }
  | { kind: "job"; job: Job }
  | { kind: "error"; message: string };

export default function App() {
  const [videos, setVideos] = useState<VideoSummary[] | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const [view, setView] = useState<View>({ kind: "home" });
  const poll = useRef<number | null>(null);

  const refresh = useCallback(() => {
    api<VideoSummary[]>("/api/videos").then(setVideos).catch(() => setVideos([]));
  }, []);
  useEffect(() => {
    refresh();
    api<Stage[]>("/api/stages").then(setStages).catch(() => {});
    return () => { if (poll.current) window.clearInterval(poll.current); };
  }, [refresh]);

  const go = useCallback((v: View) => { setView(v); window.scrollTo({ top: 0 }); }, []);

  const open = useCallback(async (id: string) => {
    go({ kind: "loading", id });
    try { setView({ kind: "report", data: await api<R>(`/api/videos/${encodeURIComponent(id)}/report`) }); }
    catch (e) { setView({ kind: "error", message: (e as Error).message }); }
  }, [go]);

  const upload = useCallback(async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    try {
      const job = await api<Job>("/api/videos", { method: "POST", body });
      go({ kind: "job", job });
      if (poll.current) window.clearInterval(poll.current);
      poll.current = window.setInterval(async () => {
        let j: Job;
        try { j = await api<Job>(`/api/jobs/${job.job_id}`); } catch { return; }
        if (j.status === "done") {
          window.clearInterval(poll.current!); poll.current = null;
          refresh();
          setView({ kind: "report", data: await api<R>(`/api/jobs/${j.job_id}/report`) });
        } else {
          if (j.status === "failed") { window.clearInterval(poll.current!); poll.current = null; }
          setView({ kind: "job", job: j });
        }
      }, 2000);
    } catch (e) { go({ kind: "error", message: `Upload failed: ${(e as Error).message}` }); }
  }, [go, refresh]);

  const back = view.kind !== "home" && (
    <button type="button" onClick={() => go({ kind: "home" })}
      className="mb-5 inline-flex items-center gap-1.5 text-[14px] text-ink-2 transition hover:-translate-x-0.5 hover:text-ink">
      <ArrowLeft weight="bold" className="size-4" />Back to the pile
    </button>
  );

  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-ink focus:px-3 focus:py-2 focus:text-bg">Skip to content</a>
      <TopBar view={view.kind} onHome={() => go({ kind: "home" })} onRatingSet={() => go({ kind: "rating" })} onUpload={upload} />
      <main id="main" className="mx-auto w-full min-w-0 max-w-[1240px] px-4 pb-20 pt-6 sm:px-8">
        {back}
        {view.kind === "home" && <Home videos={videos} onOpen={open} onUpload={upload} />}
        {view.kind === "rating" && <RatingSet onOpen={open} />}
        {view.kind === "loading" && <Spinner label="Loading the report…" />}
        {view.kind === "report" && <Report key={view.data.video_id} data={view.data} />}
        {view.kind === "job" && <JobProgress job={view.job} stages={stages} />}
        {view.kind === "error" && <Banner tone="bad" title="That didn't work">{view.message}</Banner>}
      </main>
      <footer className="border-t border-line/80 py-6 text-center font-mono text-[12px] text-mute">
        MMP-Quant · B.Tech research project · scores are the system's, not a rater's
      </footer>
    </>
  );
}
