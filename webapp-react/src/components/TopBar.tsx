import { useRef } from "react";
import { ChartBar, UploadSimple } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden>
        <rect width="32" height="32" rx="9" fill="#1d1a16" />
        <rect x="6" y="8" width="13" height="10" rx="2.5" fill="#f4efe6" />
        <rect x="8.5" y="11.2" width="8" height="1.6" rx=".8" fill="#ffe36e" />
        <path d="M8 23c3 3 5 3.6 7 3.6 4 0 8-8 9.6-15" fill="none" stroke="#e8573a" strokeWidth="2.6" strokeLinecap="round" />
      </svg>
      <span className="display hidden text-[19px] font-extrabold tracking-tight sm:inline">MMP-Quant</span>
    </span>
  );
}

type Props = { view: string; onHome: () => void; onRatingSet: () => void; onUpload: (f: File) => void };

export function TopBar({ view, onHome, onRatingSet, onUpload }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const link = (active: boolean) => cn(
    "relative rounded-lg px-3 py-2 text-[14px] font-medium transition hover:text-ink active:translate-y-px",
    active ? "text-ink after:absolute after:inset-x-3 after:-bottom-0.5 after:h-[3px] after:rounded-full after:bg-coral" : "text-ink-2");
  return (
    <header className="sticky top-0 z-30 border-b border-line/80 bg-bg/85 backdrop-blur-md">
      <nav className="mx-auto flex max-w-[1240px] items-center gap-1 px-3 py-3 sm:gap-2 sm:px-8" aria-label="Main">
        <button type="button" onClick={onHome} className="mr-1 shrink-0 rounded-lg sm:mr-2" aria-label="MMP-Quant home"><Logo /></button>
        <button type="button" onClick={onHome} className={link(view === "home")}>Library</button>
        <button type="button" onClick={onRatingSet} data-testid="nav-rating" className={cn(link(view === "rating"), "inline-flex items-center gap-1.5")}>
          <ChartBar weight="bold" className="size-4" />Rating set
        </button>
        <button type="button" onClick={() => input.current?.click()} aria-label="Mark a video (upload)"
          className="ml-auto inline-flex items-center gap-2 rounded-xl bg-ink px-3.5 py-2 text-[14px] font-semibold text-bg transition hover:bg-[#33302a] active:scale-[.98]">
          <UploadSimple weight="bold" className="size-4" /><span className="hidden sm:inline">Mark a video</span>
        </button>
        <input ref={input} type="file" hidden accept=".mp4,.mov,.mkv,.webm,.m4v,.avi"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = ""; }} />
      </nav>
    </header>
  );
}
