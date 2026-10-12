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


@app.command("config-check")
def config_check(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Validate config.yaml and print the resolved parameters."""
    cfg = Config.load(config)
    console.print(f"[green]OK[/green] {config} is valid")
    console.print_json(json.dumps(cfg.raw))


@app.command()
def validate() -> None:
    """Not implemented — build order step 4."""
    raise typer.BadParameter(
        "`validate` is not implemented yet (build order step 4). It requires "
        "data/manual_scores.csv, which does not exist. This command is left "
        "unimplemented on purpose: a stub that reads placeholder scores would "
        "produce correlation numbers that look real and are not."
    )


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
