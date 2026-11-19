"""Read-only access to a winnower catalogue.

Winnower collects and catalogues; nothing here writes to its storage. The catalogue is a
storage root (a local path, ``file://`` URL or ``s3://`` bucket) laid out the way winnower
writes it::

    samples/{camelot} {key}/{name}.flac   the loop, untouched
    samples/{camelot} {key}/{name}.json   sidecar: the source of truth
    samples/no key/{name}.flac|json       unpitched material

Sidecars are read rather than ``index/samples.jsonl`` because the index is derived and
may be stale; sidecars are what winnower itself treats as authoritative.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec


@dataclass(frozen=True)
class Sample:
    id: str
    file: str  # path relative to the catalogue root
    key: str | None  # winnower's short key, e.g. "Am"; None for unpitched material
    camelot: str | None
    bpm: float | None
    bars: int | None
    tempo_source: str | None  # "detected" (a real groove) or "fitted" (loop length only)
    duration_s: float | None
    licence: str | None
    licence_decision: str  # allow | flag | local
    attribution: str | None
    title: str | None
    source: str  # provider:provider_id, used to keep one recording from dominating a set
    centroid_hz: float | None

    @classmethod
    def from_sidecar(cls, sc: dict) -> Sample:
        key = sc.get("key") or {}
        reg = sc.get("region") or {}
        lic = sc.get("licence") or {}
        src = sc.get("source") or {}
        desc = sc.get("descriptors") or {}
        return cls(
            id=sc["id"],
            file=sc["file"],
            key=key.get("short"),
            camelot=key.get("camelot"),
            bpm=reg.get("bpm"),
            bars=reg.get("bars"),
            tempo_source=reg.get("tempo_source"),
            duration_s=(sc.get("audio") or {}).get("duration_s"),
            licence=lic.get("id"),
            licence_decision=lic.get("decision", "local"),
            attribution=src.get("attribution"),
            title=src.get("title"),
            source=f"{src.get('provider')}:{src.get('provider_id')}",
            centroid_hz=(desc.get("centroid_hz") or {}).get("mean"),
        )

    @property
    def commercial(self) -> bool:
        return self.licence_decision == "allow"


class Catalogue:
    """A winnower storage root, opened read-only."""

    def __init__(self, root: str):
        self.fs, path = fsspec.core.url_to_fs(root)
        self.root = path.rstrip("/")

    def _p(self, key: str) -> str:
        return f"{self.root}/{key.lstrip('/')}"

    def samples(self) -> Iterator[Sample]:
        prefix = self.root + "/"
        for path in sorted(self.fs.glob(self._p("samples/**/*.json"))):
            with self.fs.open(path, "rb") as f:
                sc = json.load(f)
            if "id" not in sc or "file" not in sc:
                continue
            if not sc["file"]:
                sc["file"] = path[len(prefix) :].removesuffix(".json") + ".flac"
            yield Sample.from_sidecar(sc)

    def open(self, sample: Sample):
        return self.fs.open(self._p(sample.file), "rb")

    def fingerprint(self, sample: Sample) -> str:
        """Changes when the file changes: size plus mtime locally, ETag on S3."""
        info = self.fs.info(self._p(sample.file))
        stamp = info.get("ETag") or info.get("mtime") or info.get("LastModified") or ""
        return f"{info.get('size')}-{stamp}"

    def path(self, sample: Sample) -> str:
        """A location other tools can open: a plain path locally, a URI otherwise."""
        protocols = self.fs.protocol if isinstance(self.fs.protocol, tuple) else (self.fs.protocol,)
        p = self._p(sample.file)
        return p if "file" in protocols else self.fs.unstrip_protocol(p)
