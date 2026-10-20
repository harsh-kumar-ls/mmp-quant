"""Validation harness (build order step 4): system scores vs the professor's.

Built 16 Oct 2026, before any manual scores exist, so that the method is fixed
in advance: which comparisons are made (the pre-registered variants), how the
tune / held-out split is drawn, and how uncertainty is reported. It is tested
only on synthetic score sets inside `tests/`, built to have known answers --
never on placeholder files in `data/`.
"""
