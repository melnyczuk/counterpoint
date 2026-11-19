"""``counterpoint`` command line.

    counterpoint match --chord Am9
    counterpoint match --midi phrase.mid --set 4 --texture
    counterpoint match --audio take.wav --scale "A dorian" --json
    counterpoint match --chord Fmaj7#11 --bpm 96 --export ~/Music/sets/fmaj
    counterpoint profile                      # analyse the catalogue ahead of time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from . import config
from .catalogue import Catalogue
from .harmony import Target, parse_chord
from .profile import ProfileCache, profiles_for
from .scoring import Match
from .select import Filters, layering_set, rank


def _bpm_range(text: str) -> tuple[float, float]:
    lo, _, hi = text.partition("-")
    return float(lo), float(hi or lo)


def _progress(done: int, total: int) -> None:
    if sys.stderr.isatty() or done == total or done % 25 == 0:
        print(f"\ranalysing {done}/{total}", end="" if done < total else "\n", file=sys.stderr)


def _open(args) -> tuple[config.Settings, Catalogue, list, dict]:
    settings = config.load(args.config)
    root = args.catalogue or settings.catalogue
    if not root:
        sys.exit("no catalogue: pass --catalogue, set it in counterpoint.toml, "
                 "or set WINNOWER_STORAGE_ROOT")
    cat = Catalogue(root)
    samples = list(cat.samples())
    cache = ProfileCache(settings.cache_dir / "profiles.json")
    profiles = profiles_for(cat, samples, cache, _progress)
    return settings, cat, samples, profiles


def _target(args) -> Target:
    sources = [x for x in (args.chord, args.midi, args.audio) if x]
    if len(sources) != 1:
        sys.exit("give exactly one of --chord, --midi or --audio")
    if args.chord:
        return Target.from_chord(parse_chord(args.chord), args.scale, args.bpm)
    from .inputs import target_from_audio, target_from_midi

    if args.midi:
        return target_from_midi(args.midi, args.scale, args.bpm)
    return target_from_audio(args.audio, args.scale, args.bpm)


def _row(m: Match) -> str:
    s = m.sample
    st = m.stretch
    stretch = ""
    if st is not None:
        stretch = f"x{st.ratio:.3f}" + ("" if st.time == "normal" else f" {st.time}")
    shift = f"{m.shift:+d}" if m.shift else "0"
    key = s.key or "-"
    flag = "" if s.commercial else " *"
    return (f"{m.score:6.3f}  {shift:>3}  {key:<4} {s.bpm or '-':>5}  {stretch:<14} "
            f"fit {m.parts.fit:.2f} col {m.parts.colour:.2f} rough {m.parts.roughness:.2f}  "
            f"{Path(s.file).name}{flag}")


def _export(cat: Catalogue, target: Target, matches: list[Match], out: Path) -> None:
    """A set folder of symlinks to the untouched files, named with the shift to apply."""
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for m in matches:
        src = cat.path(m.sample)
        prefix = f"{m.shift:+d}" if m.shift else "0"
        name = f"{prefix}_{Path(m.sample.file).name}"
        dest = out / name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        try:
            os.symlink(src, dest)
        except OSError:
            import shutil

            with cat.open(m.sample) as f, open(dest, "wb") as g:
                shutil.copyfileobj(f, g)
        entries.append({"link": name, "source": src, **m.to_dict(),
                        "attribution": m.sample.attribution})
    (out / "set.json").write_text(json.dumps(
        {"target": target.label, "scale": target.scale_label, "bpm": target.bpm,
         "samples": entries}, indent=2, ensure_ascii=False))


def _match(args) -> None:
    settings, cat, samples, profiles = _open(args)
    target = _target(args)
    limits = settings.limits
    if args.max_shift is not None:
        limits = replace(limits, max_shift=args.max_shift)
    if args.no_shift:
        limits = replace(limits, max_shift=0)
    if args.max_stretch is not None:
        limits = replace(limits, max_stretch=args.max_stretch)
    if args.no_stretch:
        limits = replace(limits, stretch=False)
    filters = Filters(commercial=args.commercial,
                      bpm_range=args.bpm_filter if not target.bpm else None)
    pitched, textures = rank(samples, profiles, target, settings.weights, limits, filters)
    layered = (layering_set(pitched, target, settings.weights, args.set, textures, args.texture)
               if args.set else [])
    top = pitched[: args.limit]

    if args.json:
        print(json.dumps({
            "target": target.label, "scale": target.scale_label, "bpm": target.bpm,
            "matches": [m.to_dict() for m in top],
            "set": [m.to_dict() for m in layered],
            "textures": [m.to_dict() for m in textures[: args.limit]] if args.texture else [],
        }, indent=2, ensure_ascii=False))
    else:
        tempo = f", {target.bpm} BPM" if target.bpm else ""
        print(f"{target.label} in {target.scale_label}{tempo}: "
              f"{len(pitched)} pitched matches of {len(samples)} samples\n")
        print(" score  shift key    bpm  stretch")
        for m in top:
            print(_row(m))
        if layered:
            print(f"\nlayering set of {len(layered)}:")
            for m in layered:
                print(_row(m))
        if any(not m.sample.commercial for m in top + layered):
            print("\n* licence flagged (NC, SA or unknown): clear it before release")
    if args.export:
        chosen = layered or top
        _export(cat, target, chosen, Path(args.export).expanduser())
        print(f"\nexported {len(chosen)} to {args.export}", file=sys.stderr)


def _profile(args) -> None:
    _, _, samples, profiles = _open(args)
    pitched = sum(p.pitched for p in profiles.values())
    print(f"{len(samples)} samples profiled: {pitched} pitched, {len(samples) - pitched} unpitched")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="counterpoint", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="settings file (default ./counterpoint.toml)")
    ap.add_argument("--catalogue", help="winnower storage root")
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("match", help="rank samples against a chord, MIDI or audio")
    m.add_argument("--chord", help="chord symbol, e.g. Am9, Fmaj7#11, C/G")
    m.add_argument("--midi", help="MIDI file whose notes form the target")
    m.add_argument("--audio", help="audio clip whose pitches (and tempo) form the target")
    m.add_argument("--scale", help='scale for context, e.g. "A dorian" (default: inferred)')
    m.add_argument("--bpm", type=float, help="target tempo (overrides one from MIDI/audio)")
    m.add_argument("--max-shift", type=int, help="largest suggested shift in semitones")
    m.add_argument("--no-shift", action="store_true", help="only samples that fit unshifted")
    m.add_argument("--max-stretch", type=float, help="largest stretch ratio, e.g. 1.25")
    m.add_argument("--no-stretch", action="store_true",
                   help="don't suggest stretches; use the target tempo as a filter")
    m.add_argument("--bpm-filter", type=_bpm_range, help="with no target tempo: 96 or 90-110")
    m.add_argument("--commercial", action="store_true", help="only allow-listed licences")
    m.add_argument("--limit", type=int, default=20, help="matches to show")
    m.add_argument("--set", type=int, default=3, help="layering set size (0 for none)")
    m.add_argument("--texture", action="store_true", help="add an unpitched texture layer")
    m.add_argument("--json", action="store_true")
    m.add_argument("--export", help="write a set folder of symlinks plus set.json here")
    m.set_defaults(func=_match)

    p = sub.add_parser("profile", help="analyse the catalogue ahead of time")
    p.set_defaults(func=_profile)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as e:
        sys.exit(str(e))
