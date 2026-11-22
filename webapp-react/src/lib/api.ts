/* Types for the FastAPI JSON (src/mmpquant/report_data.py, webapp/server.py).
   Only the fields this UI reads are typed. */

export type VideoSummary = { video_id: string; score: number | null; status: string; duration: number; n_segments: number; style?: string | null; coherence?: number | null; instructor?: number | null };

export type SentenceCheck = {
  sentence: string; narration: string; narration_from: number | null; narration_to: number | null;
  spans: [number, number][]; visible_s: number; is_question: boolean;
  content_words: string[]; shared_words: string[]; coverage: number; chance: number; passed: boolean;
  verbatim_raw?: number; verbatim_chance?: number; verbatim?: number;
  semantic_raw?: number; semantic_chance?: number; semantic?: number;
  score?: number | null; time_weight?: number;
};

export type WindowBreakdown = {
  index: number; start: number; end: number; duration: number; timecode: string;
  qualifies: boolean; exclusion_reason: string | null; exclusion_label: string | null;
  score: number | null; gate: string | null; onscreen_compared: string; labels_excluded: string[];
  n_labels_excluded: number; narration: string;
  narration_timed: { start: number; end: number; text: string; in_window: boolean }[];
  sentence_checks: SentenceCheck[]; null_windows: number | null;
  verbatim: number | null; semantic: number | null; screen_time_share: number | null;
};

export type Segment = {
  index: number; timecode: string; start: number; end: number; score: number | null;
  windows: WindowBreakdown[];
};

export type CoherenceWindow = {
  index: number; start: number; end: number; duration: number; timecode: string; qualifies: boolean;
  score: number | null; narration: string; background_music: number; off_topic: number;
  narrated_s: number; music_under_speech_s: number;
};

export type Coherence = {
  score: number | null; status: string; na_reason: string | null;
  parts: { background_music: number; off_topic: number };
  alternative?: { weights: Record<string, number>; score: number | null };
  music_loudness: { median_db: number | null; n_seconds: number };
  windows: CoherenceWindow[];
  segments: { index: number; timecode: string; start: number; end: number; score: number | null; parts: Record<string, number> }[];
  methodology: { weights: { background_music: number; off_topic: number }; music_threshold: number; audio_model: string };
};

export type Methodology = {
  weight_verbatim: number; weight_semantic: number; min_onscreen_words: number; embedding_model: string;
  window_size_s: number; segment_length_s: number; min_sentence_coverage: number;
  min_shared_content_words: number; null_exclusion_s: number; alignment_tolerance_s: number;
  nearby_reach_s: number | null; frame_fps: number; asr_backend: string; ocr_backend: string;
};

export type Report = {
  video_id: string; duration: number; score: number | null; assessed_score: number | null;
  coverage: number | null; status: string; na_reason: string | null; warnings: string[];
  n_windows_total: number; n_windows_qualifying: number;
  segments: Segment[]; coherence: Coherence | null; methodology: Methodology;
  instructor?: { score: number | null; formula: string; segments: { index: number; timecode: string; score: number | null }[] } | null;
  worst_segments: { segment_index: number; timecode: string; score: number;
    windows: { timecode: string; score: number; onscreen_text: string; narration_text: string; restated_sentences: string[] }[] }[];
};

export type RatingRow = {
  target: number; video_id: string; style: string; duration_s: number | null;
  redundancy: number | null; distance: number | null; coherence: number | null;
  current: { redundancy: number | null; coherence: number | null } | null;
};
export type RatingSet = { package: string; rows: RatingRow[]; total_distance: number };

export type Stage = { key: string; label: string; description: string };
export type Job = { job_id: string; video_id: string; status: string; stage: string; stage_index: number; error: string | null };

// Read-only copy (13 Nov 2026, tools/build_static_site.py): no server; every
// GET reads a pre-built JSON file, and nothing can be uploaded.
export const STATIC = import.meta.env.VITE_STATIC === "1";

function staticPath(path: string): string {
  const base = import.meta.env.BASE_URL;
  if (path === "/api/videos") return `${base}data/videos.json`;
  if (path === "/api/stages") return `${base}data/stages.json`;
  const m = path.match(/^\/api\/videos\/([^/]+)\/report/);
  if (m) return `${base}data/reports/${encodeURIComponent(decodeURIComponent(m[1]))}.json`;
  throw new Error("Not available in the read-only copy.");
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (STATIC && init?.method && init.method !== "GET") throw new Error("Uploading is not available in the read-only copy.");
  const r = await fetch(STATIC ? staticPath(path) : path, STATIC ? undefined : init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = String(j.detail); } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}
