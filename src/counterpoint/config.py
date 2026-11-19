"""Settings from ``counterpoint.toml`` (all optional), overridden by flags.

```toml
catalogue = "/home/pi/winnower/data"   # a winnower storage root; s3:// works with the s3 extra
cache_dir = "~/.cache/counterpoint"

[limits]
max_shift = 3
max_stretch = 1.25

[weights]
fit = 1.0
colour = 0.3
colour_cap = 0.25
roughness = 1.0
shift = 0.02
stretch = 0.4
stretch_fitted = 0.1
masking = 0.3
same_source = 0.5
```

``WINNOWER_STORAGE_ROOT`` is used for the catalogue when nothing else names one, so a
machine already set up for winnower needs no extra setting.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .profile import default_cache_dir
from .scoring import Limits, Weights


@dataclass
class Settings:
    catalogue: str | None = None
    cache_dir: Path = field(default_factory=default_cache_dir)
    weights: Weights = field(default_factory=Weights)
    limits: Limits = field(default_factory=Limits)


def load(path: str | None = None) -> Settings:
    candidates = [Path(path)] if path else [Path("counterpoint.toml")]
    raw: dict = {}
    for p in candidates:
        if p.exists():
            raw = tomllib.loads(p.read_text())
            break
        if path:
            raise FileNotFoundError(path)
    s = Settings()
    s.catalogue = raw.get("catalogue") or os.environ.get("WINNOWER_STORAGE_ROOT")
    if raw.get("cache_dir"):
        s.cache_dir = Path(raw["cache_dir"]).expanduser()
    s.weights = Weights.from_dict(raw.get("weights", {}))
    lim = raw.get("limits", {})
    s.limits = Limits(
        max_shift=int(lim.get("max_shift", Limits.max_shift)),
        max_stretch=float(lim.get("max_stretch", Limits.max_stretch)),
    )
    return s
