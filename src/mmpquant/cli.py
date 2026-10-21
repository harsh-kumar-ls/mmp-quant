"""Command-line interface.

Implemented so far (build order steps 1-3):
    mmpquant extract <video>       run the extraction layer, dump intermediates
    mmpquant score <video>         extract + score, write result.json + report.md
    mmpquant report <run_dir>      re-render report.md without re-running ASR/OCR
    mmpquant annotation-sheet      write a blank per-segment sheet for a rater
    mmpquant config-check          validate config.yaml and show the resolved values
    mmpquant serve                 local web UI over the same pipeline (needs
                                    the `webapp` extra — see mmpquant.webapp)

NOT implemented yet, deliberately, because the build order says so:
    `mmpquant validate` (step 4). It must not exist as a stub — a command that
    appears to work but reads no real ground truth is how fabricated numbers
    end up in a thesis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Config
from .pipeline import run_pipeline
from .report import render_markdown
from .segments import write_annotation_sheet
from .stages import run_extraction

app = typer.Typer(add_completion=False, help="Quantify Mayer's Redundancy Principle.")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _run_dir(video: Path, output_root: Path, run_name: str | None) -> Path:
    return output_root / (run_name or video.stem)


@app.command()
def extract(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    output: Path | None = typer.Option(None, "--output", "-o", help="run directory"),
    no_resume: bool = typer.Option(False, "--no-resume", help="recompute ASR and OCR"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run ASR + OCR + alignment and write every intermediate for inspection."""
    _setup_logging(verbose)
    cfg = Config.load(config)
    outdir = output or _run_dir(video, Path(cfg.get("output.dir", "runs")), None)

    result = run_extraction(video, outdir, cfg, resume=not no_resume)

    table = Table(title=f"Extraction — {video.name}", show_header=False)
    table.add_row("duration", f"{result.duration:.1f}s")
    table.add_row("transcript language", str(result.transcript.language))
    table.add_row("transcript words", str(len(result.transcript.words)))
    table.add_row("OCR blocks", str(len(result.ocr_blocks)))
    table.add_row("windows", str(len(result.windows)))
    table.add_row("artifacts", str(outdir))
    console.print(table)

    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


@app.command()
def score(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    output: Path | None = typer.Option(None, "--output", "-o", help="run directory"),
    no_resume: bool = typer.Option(False, "--no-resume", help="recompute ASR and OCR"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score one video (redundancy, and coherence if enabled) and write result.json."""
    _setup_logging(verbose)
    cfg = Config.load(config)
    outdir = output or _run_dir(video, Path(cfg.get("output.dir", "runs")), None)

    pipeline = run_pipeline(video, outdir, cfg, no_resume=no_resume)
    extraction, result, segments = pipeline.extraction, pipeline.result, pipeline.segments
    segment_length = pipeline.payload["segment_length_s"]

    if result.score is None:
        console.print(f"[bold yellow]redundancy: N/A[/bold yellow] — {result.na_reason}")
        console.print("[dim]N/A is not zero. Do not substitute a number for this.[/dim]")
    else:
        console.print(f"[bold]redundancy: {result.score:.2f} / 100[/bold]")
        console.print(
            f"[dim]{result.n_windows_qualifying} of {result.n_windows_total} "
            f"windows qualified[/dim]"
        )

    rel = extraction.ocr_reliability.get("reliability")
    if rel == "text_present_but_unreadable":
        console.print()
        console.print(
            "[bold red]OCR UNRELIABLE — this score is not trustworthy[/bold red]"
        )
        console.print(f"[red]{extraction.ocr_reliability.get('note', '')}[/red]")
        console.print()

    if segments:
        table = Table(title=f"Per-segment scores ({segment_length:.0f}s segments)")
        table.add_column("segment")
        table.add_column("time")
        table.add_column("score", justify="right")
        table.add_column("windows scored", justify="right")
        for seg in segments:
            table.add_row(
                str(seg.index),
                seg.timecode,
                "N/A" if seg.score is None else f"{seg.score:.2f}",
                f"{seg.n_windows_qualifying}/{seg.n_windows}",
            )
        console.print(table)

    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    coh = pipeline.payload.get("coherence")
    if coh is not None:
        console.print()
        cres = coh["result"]
        if cres["score"] is None:
            console.print(f"[bold yellow]coherence: N/A[/bold yellow] — {cres['na_reason']}")
        else:
            parts = coh["parts"]
            console.print(f"[bold]coherence: {cres['score']:.2f} / 100[/bold]  "
                          f"[dim](background music {parts['background_music']:.2f}, "
                          f"off-topic narration {parts['off_topic']:.2f})[/dim]")
        for warning in cres["warnings"]:
            console.print(f"[yellow]coherence warning:[/yellow] {warning}")

    console.print(f"[dim]artifacts: {outdir}[/dim]")
    console.print(f"[dim]report:    {outdir / 'report.md'}[/dim]")


@app.command()
def report(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False,
                                   help="a run directory containing result.json"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    top: int = typer.Option(3, "--top", help="how many worst segments to detail"),
) -> None:
    """Re-render the Markdown report from an existing run.

    `score` already writes one. This exists so the report can be regenerated
    after the template changes without paying for ASR and OCR again, and so a
    run produced before the generator existed can still be written up.
    """
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise typer.BadParameter(f"{result_path} not found — is this a run directory?")

    payload = json.loads(result_path.read_text())
    target = output or run_dir / "report.md"
    target.write_text(render_markdown(payload, top_n=top), encoding="utf-8")
    console.print(f"[green]wrote[/green] {target}")


@app.command("annotation-sheet")
def annotation_sheet(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Write a blank per-segment sheet for human annotation.

    Contains timestamps only, never system scores: a rater shown the machine's
    answer anchors to it, and the resulting correlation would measure agreement
    with that anchor rather than independent judgement.
    """
    from .extract import media

    cfg = Config.load(config)
    duration = media.duration_seconds(video)
    length_s = float(cfg.get("segments.length_s"))
    raters = list(cfg.get("segments.raters", ["rater_1", "rater_2"]))
    target = output or Path("data/annotations") / f"{video.stem}_sheet.csv"

    path = write_annotation_sheet(video.stem, duration, length_s, target, raters)
    n = len(path.read_text().splitlines()) - 1
    console.print(f"[green]wrote[/green] {path}")
    console.print(
        f"[dim]{n} segments of {length_s:.0f}s for {duration / 60:.1f} min of video, "
        f"columns for: {', '.join(raters)}[/dim]"
    )
    console.print(
        "[dim]Send this to each rater separately. Do not score the video yourself "
        "first — blind annotation is what makes the correlation meaningful.[/dim]"
    )


@app.command("rating-workbook")
def rating_workbook(
    output: Path = typer.Option(Path("rating_package/ratings_template.xlsx"), "--output", "-o"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    scale_min: int = typer.Option(1, "--scale-min"),
    scale_max: int = typer.Option(5, "--scale-max"),
    selection: Path | None = typer.Option(
        None, "--selection", help="only the videos in this selection table (select-for-rating)"),
) -> None:
    """Write the blank rating workbook for raters: every 30 s segment of every
    video in data/videos (or only the selected ones), plus one row per video.
    Carries no system scores."""
    from .extract import media
    from .rating_workbook import build_workbook

    cfg = Config.load(config)
    exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    keep = None
    if selection is not None:
        import csv as _csv

        keep = {r["video_id"] for r in _csv.DictReader(selection.open())}
    vids = [(p.stem, media.duration_seconds(p)) for p in sorted(Path("data/videos").iterdir())
            if p.suffix.lower() in exts and (keep is None or p.stem in keep)]
    if keep is not None and len(vids) != len(keep):
        raise typer.BadParameter(f"selected videos missing from data/videos: "
                                 f"{sorted(keep - {v for v, _ in vids})}")
    out = build_workbook(vids, output, segment_length_s=float(cfg.get("segments.length_s")),
                         scale=(scale_min, scale_max))
    console.print(f"[green]wrote[/green] {out} — {len(vids)} videos, scale "
                  f"{scale_min}-{scale_max}, no system scores")


@app.command("select-for-rating")
def select_for_rating(
    output: Path = typer.Option(Path("rating_package/selection.csv"), "--output", "-o"),
    max_duration: float = typer.Option(200.0, "--max-duration", help="seconds (3:20)"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Choose the 10 videos for the professor by the pre-registered rule
    (PROGRESS.md, 20 Oct 2026). The table holds system scores: student only."""
    from .selection import load_candidates, load_styles, select, write_table

    cfg = Config.load(config)
    exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    videos = [p.stem for p in sorted(Path("data/videos").iterdir()) if p.suffix.lower() in exts]
    cands, skipped = load_candidates(Path(str(cfg.get("output.dir", "runs"))), videos,
                                     load_styles(Path("data/video_styles.csv")), max_duration)
    sel = select(cands)
    table = Table(title=f"Selected for rating ({len(cands)} eligible, total distance "
                        f"{sel.cost:.1f})")
    for col in ("target", "video", "style", "length", "redundancy", "coherence", "loudness"):
        table.add_column(col)
    for t, c in sel.slots:
        table.add_row(str(t), c.video_id, c.style,
                      f"{int(c.duration_s) // 60}:{int(c.duration_s) % 60:02d}",
                      f"{c.redundancy:.2f}", f"{c.coherence:.2f}",
                      "" if c.loudness_db is None else f"{c.loudness_db:.1f} dB")
    console.print(table)
    for line in sel.swaps:
        console.print(f"[dim]swap: {line}[/dim]")
    for d in sel.unmet:
        console.print(f"[bold yellow]constraint not met:[/bold yellow] {d}")
    for v, why in skipped.items():
        console.print(f"[dim]not eligible: {v} — {why}[/dim]")
    console.print(f"[green]wrote[/green] {write_table(sel, output)} "
                  "[dim](has system scores — do not send to the professor)[/dim]")


@app.command("config-check")
def config_check(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Validate config.yaml and print the resolved parameters."""
    cfg = Config.load(config)
    console.print(f"[green]OK[/green] {config} is valid")
    console.print_json(json.dumps(cfg.raw))


@app.command()
def validate(
    principle: str = typer.Option("all", "--principle", "-p",
                                  help="redundancy, coherence, or all"),
    ground_truth: Path | None = typer.Option(
        None, "--ground-truth", "-g",
        help="manual scores: a CSV or a folder of CSVs (default: from config.yaml)"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    allow_mixed_config: bool = typer.Option(
        False, "--allow-mixed-config",
        help="proceed even if some runs were scored with a different config (recorded)"),
    level: str = typer.Option(
        "segment", "--level",
        help="for rating workbooks: 'segment' (Ratings sheet) or 'video' (Video overall)"),
) -> None:
    """Compare system scores with the professor's manual scores.

    Reads each video's existing run (score videos first). Writes a report,
    validation.json, residuals.csv and a scatter plot per principle.
    """
    from datetime import datetime

    from .validation.ground_truth import GroundTruthError, load_ground_truth
    from .validation.harness import Settings, ValidationError, run_validation

    cfg = Config.load(config)
    principles = ["redundancy", "coherence"] if principle == "all" else [principle]
    unknown = set(principles) - {"redundancy", "coherence"}
    if unknown:
        raise typer.BadParameter(f"unknown principle(s): {sorted(unknown)}")
    settings = Settings(
        tune_fraction=float(cfg.get("validation.tune_fraction", 0.6)),
        seed=int(cfg.get("validation.seed", 0)),
        n_resamples=int(cfg.get("validation.n_resamples", 10000)),
        allow_mixed_config=allow_mixed_config,
    )
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    failed = False
    for name in principles:
        path = ground_truth or Path(str(cfg.get(f"validation.ground_truth.{name}")))
        if not path.exists():
            console.print(
                f"[yellow]{name}: no manual scores at {path}[/yellow] — validation needs the "
                "professor's scores. Nothing is computed from placeholder data.")
            failed = True
            continue
        try:
            gt = load_ground_truth(path, principle=name, level=level)
            out = Path(str(cfg.get("validation.out_dir", "runs/validation"))) / f"{name}-{stamp}"
            r = run_validation(name, gt, Path(str(cfg.get("output.dir", "runs"))), out,
                               current_config=cfg.raw, settings=settings)
        except (GroundTruthError, ValidationError) as exc:
            console.print(f"[bold red]{name}: {exc}[/bold red]")
            failed = True
            continue
        if r["is_dummy_ground_truth"]:
            console.print("[bold red]TEST DATA — NOT A RESULT: the manual-score file is "
                          "marked dummy/synthetic.[/bold red]")
        h, loo = r["held_out"], r["nested_loo"]
        console.print(
            f"[bold]{name}[/bold] ({r['level']} level): held-out Spearman "
            f"[bold]{h['spearman']}[/bold] (95% CI {h['spearman_ci95'][0]} to "
            f"{h['spearman_ci95'][1]}, {h['n_videos']} videos, variant {h['variant']}); "
            f"leave-one-video-out {loo['spearman']}")
        console.print(f"[dim]report: {out / 'report.md'}[/dim]")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def regress(
    from_runs: bool = typer.Option(
        False, "--from-runs", help="check the existing runs/ as they are (no re-scoring)"),
    accept: bool = typer.Option(
        False, "--accept",
        help="make these scores the accepted baseline (only after reviewing the changes)"),
    videos: list[str] = typer.Option(None, "--video", "-v", help="limit to these videos"),
    use_last: bool = typer.Option(
        False, "--use-last",
        help="use the results of the last re-score (runs/_regress) instead of re-scoring"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Re-score every video and check it against the user's expectations and
    the last accepted scores. Exit 0 = all clear, 1 = an expectation failed or
    a video could not be scored, 2 = scores changed and need review."""
    from . import regression as rg

    _setup_logging(False)
    cfg = Config.load(config)
    runs_dir = Path(str(cfg.get("output.dir", "runs")))
    scratch_root = runs_dir / "_regress"
    scratch = scratch_root / "runs"
    found = {p.stem: p for p in sorted(Path("data/videos").iterdir())
             if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}}
    if videos:
        found = {k: v for k, v in found.items() if k in set(videos)}
    problems: dict[str, str] = {}
    if from_runs:
        results = rg.load_results(runs_dir, list(found))
        problems = {v: "no run" for v in found if v not in results}
    elif use_last:
        # Scoring is reproducible (20 Oct: a full re-score matched the baseline
        # exactly), so the last re-score can be accepted without repeating it.
        results = rg.load_results(scratch, list(found))
        problems = {v: "not in the last re-score" for v in found if v not in results}
    else:
        console.print(f"re-scoring {len(found)} videos from cached extraction "
                      f"(one process, models loaded once) into {scratch} …")
        problems = rg.rescore_all(found, runs_dir, scratch, cfg,
                                  log=lambda m: console.print(f"[dim]{m}[/dim]"))
        results = rg.load_results(scratch, list(found))

    checks = rg.check_expectations(results, only=set(found) if videos else None)
    current = rg.snapshot(results)
    baseline = json.loads(rg.BASELINE.read_text()) if rg.BASELINE.exists() else None
    if baseline is not None and videos:
        baseline = {k: v for k, v in baseline.items() if k in current or k in set(videos)}
    changes = rg.diff(baseline, current) if baseline is not None else []
    scratch_root.mkdir(parents=True, exist_ok=True)
    report = scratch_root / "report.md"
    report.write_text(rg.render(checks, changes, problems, current, baseline is not None))

    failed = [c for c in checks if not c.passed]
    for c in checks:
        mark = "[green]PASS[/green]" if c.passed else "[bold red]FAIL[/bold red]"
        console.print(f"{mark} {c.id} [dim]({c.status})[/dim] {c.detail}")
    for v, why in problems.items():
        console.print(f"[yellow]not scored[/yellow] {v}: {why}")
    if baseline is None:
        console.print("[yellow]no accepted baseline yet[/yellow]")
    elif changes:
        console.print(f"[bold yellow]{len(changes)} score change(s) vs the accepted "
                      f"baseline[/bold yellow] — review {report}")
    else:
        console.print("[green]no score changes vs the accepted baseline[/green]")

    if accept:
        if failed or problems:
            console.print("[bold red]not accepted:[/bold red] expectations fail or videos "
                          "were not scored — fix those first.")
            raise typer.Exit(code=1)
        if not from_runs:
            rg.accept(scratch, runs_dir, list(results))
        merged = {**(json.loads(rg.BASELINE.read_text()) if rg.BASELINE.exists() else {}),
                  **current}
        rg.BASELINE.write_text(json.dumps(merged, indent=1))
        console.print(f"[green]accepted[/green]: baseline updated ({len(current)} videos)")
        raise typer.Exit(code=0)
    console.print(f"[dim]report: {report}[/dim]")
    if failed or problems:
        raise typer.Exit(code=1)
    if changes:
        raise typer.Exit(code=2)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Run the local web UI: upload a video, watch it score, read the report.

    A thin presentation layer over the same pipeline `score` uses — every
    number it shows lands in `runs/<video_id>/` exactly as a CLI run would.
    Requires the `webapp` extra (`uv pip install -e ".[webapp]"`).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "the web UI needs its extra dependencies: "
            'uv pip install -e ".[webapp]"'
        ) from exc

    console.print(f"[bold]MMP-Quant[/bold] — open [cyan]http://{host}:{port}[/cyan]")
    console.print("[dim]Ctrl+C to stop[/dim]")
    uvicorn.run("mmpquant.webapp.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
