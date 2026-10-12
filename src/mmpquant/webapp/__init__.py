"""Local web UI: upload a video, watch it score, read the structured report.

This package is a thin presentation layer over the same pipeline the CLI uses
(`mmpquant.pipeline.run_pipeline`). It recomputes nothing and defines no new
scoring behaviour — every number a browser ever shows came from the identical
code path `mmpquant score` runs, and lands in the same `runs/<video_id>/`
directory. Uninstalling this package changes nothing about how a video is
scored.
"""
