"""Config loading.

One YAML file is the single source of truth for every tunable parameter. The
loaded dict is carried through the run and dumped verbatim into the report's
methodology appendix, so every number in the thesis is traceable to the exact
config that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

_MISSING = object()


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls(raw=raw, source_path=p)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        cfg = cls(raw=raw)
        cfg.validate()
        return cfg

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node

    # ----------------------------------------------------------------- #
    # Validation: fail loudly at load time rather than producing a
    # quietly-wrong score 20 minutes into a run.
    # ----------------------------------------------------------------- #
    def validate(self) -> None:
        w = self.get("scoring.redundancy.weights")
        if not isinstance(w, dict) or {"verbatim", "semantic"} - w.keys():
            raise ConfigError("scoring.redundancy.weights needs 'verbatim' and 'semantic'")
        total = float(w["verbatim"]) + float(w["semantic"])
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"scoring.redundancy.weights must sum to 1.0, got {total}. "
                "Non-unit weights silently change the score range and make runs "
                "incomparable."
            )
        for key in ("verbatim", "semantic"):
            if float(w[key]) < 0:
                raise ConfigError(f"weight '{key}' must be >= 0")

        if float(self.get("windowing.size_s")) <= 0:
            raise ConfigError("windowing.size_s must be > 0")
        if float(self.get("alignment.tolerance_s")) < 0:
            raise ConfigError("alignment.tolerance_s must be >= 0")
        if self.get("alignment.expand") not in ("asr", "ocr", "both"):
            raise ConfigError("alignment.expand must be one of: asr, ocr, both")
        if float(self.get("media.frame_fps")) <= 0:
            raise ConfigError("media.frame_fps must be > 0")
        if int(self.get("scoring.redundancy.min_onscreen_words")) < 0:
            raise ConfigError("scoring.redundancy.min_onscreen_words must be >= 0")

        principles = self.get("scoring.principles", ["redundancy"])
        unknown = set(principles) - {"redundancy", "coherence"}
        if unknown:
            raise ConfigError(f"scoring.principles: unknown principle(s) {sorted(unknown)}")
        if "coherence" in principles:
            cw = self.get("scoring.coherence.weights")
            if not isinstance(cw, dict) or {"background_music", "off_topic"} - cw.keys():
                raise ConfigError(
                    "scoring.coherence.weights needs 'background_music' and 'off_topic'")
            if any(float(v) < 0 for v in cw.values()):
                raise ConfigError("scoring.coherence.weights must be >= 0")
            if abs(sum(float(v) for v in cw.values()) - 1.0) > 1e-6:
                raise ConfigError("scoring.coherence.weights must sum to 1.0")
            aw = self.get("scoring.coherence.alternative_weights", None)
            bad_alt = aw is not None and (
                set(aw) != {"background_music", "off_topic"}
                or any(float(v) < 0 for v in aw.values())
                or abs(sum(float(v) for v in aw.values()) - 1.0) > 1e-6
            )
            if bad_alt:
                raise ConfigError("scoring.coherence.alternative_weights: background_music and "
                                  "off_topic, >= 0, summing to 1.0")
            t = float(self.get("scoring.coherence.background_music.threshold"))
            if not 0.0 < t < 1.0:
                raise ConfigError("scoring.coherence.background_music.threshold must be in (0, 1)")
            d = float(self.get("scoring.coherence.off_topic.decision_point"))
            if not 0.0 < d < 1.0:
                raise ConfigError("scoring.coherence.off_topic.decision_point must be in (0, 1)")
