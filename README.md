# counterpoint

Finds samples in a [winnower](https://github.com/melnyczuk/winnower) catalogue that sit
harmonically against a chord, a MIDI phrase or an audio clip. For each one it suggests
the transposition and time-stretch that make it fit, and it picks small sets that layer
well together.

It scores each sample's 12-note pitch profile against the target at every
transposition: how much of it is in the scale (fit), how much adds colour without
reharmonising (9ths, 11ths, 13ths), and how rough it is against the chord tones. Every
part of the score is reported, so a match can be explained and the weights tuned by ear.

**Nothing is ever changed on disk.** Winnower collects and catalogues, and Counterpoint
only reads its storage. Shifts and stretches are suggestions you apply at use time (an
Ableton clip's Transpose and Warp), so a sample can be reused at different shifts in
different places.

Design and roadmap: [docs/design.md](docs/design.md). That covers the planned `pack`
command (a key-sorted folder of samples for Ableton), ensemble `voice` mode (instances
that listen to each other and change sample every few loops) and running one voice per
Raspberry Pi in a spatial installation.

## Status (v0.1)

- [x] Read-only winnower catalogue (local path or `s3://` with the `s3` extra)
- [x] Pitch profiles (chroma, pitchedness, band shares), cached outside winnower's storage
- [x] Targets from chord symbols, MIDI files and audio clips, with inferred or given scale
- [x] Scoring at all 12 transpositions: fit, colour, roughness, shift cost
- [x] Tempo matching: stretch ratio with half/double time, stretch cost, limits
- [x] Layering sets: combined fit, band masking and same-recording penalties, texture layer
- [x] `counterpoint match` with table, JSON and an Ableton set-folder export
- [ ] `counterpoint pack`
- [ ] Ensemble `voice` mode, Link, OSC, playback
- [ ] Max for Live device
- [ ] Raspberry Pi deployment

## Install

```sh
uv sync                 # add --extra s3 to read a winnower bucket
. .venv/bin/activate    # or prefix commands with `uv run`
```

Point it at a winnower storage root with `--catalogue`, `catalogue = "..."` in
`counterpoint.toml`, or `WINNOWER_STORAGE_ROOT` (which winnower already uses).

## Use

```sh
counterpoint --catalogue ~/winnower/data profile          # analyse ahead of time (cached)
counterpoint match --chord Am9                            # top matches and a layering set of 3
counterpoint match --chord Fmaj7#11 --bpm 96 --set 4 --texture
counterpoint match --midi phrase.mid --scale "D dorian"   # tempo from the file's tempo map
counterpoint match --audio take.wav --json                # pitches and tempo from a recording
counterpoint match --chord C/G --no-shift --commercial    # as-is fits, release-safe licences
counterpoint match --chord Em --export ~/Music/sets/em    # symlinks named +2_..., plus set.json
```

Example output:

```
Am9 in A minor, 96.0 BPM: 41 pitched matches of 230 samples

 score  shift key    bpm  stretch
 0.962    0  Am      96                 fit 0.98 col 0.21 rough 0.03  Am_96bpm_4bar_...flac
 0.911   +2  Gm     100  x0.960         fit 0.95 col 0.12 rough 0.05  Gm_100bpm_2bar_...flac
```

- **shift**: semitones to put in the clip's Transpose.
- **stretch**: playback speed to reach the target tempo. `half` or `double` means it fits
  better at half or double time, so it's warped as such.
- `*` after a file name means its licence is flagged (NC, SA or unknown): clear it before
  release.

Chord symbols: `C`, `Am`, `F#m7`, `Bbmaj7`, `Fmaj7#11`, `G7b9`, `Bø`, `Bm7b5`, `Cdim7`,
`Caug`, `Dsus4`, `Esus2`, `Cadd9`, `C6/9`, `E5`, `G7alt`, `C/G`.

## Settings

Everything is optional. See `src/counterpoint/config.py` for the full `counterpoint.toml`,
including the scoring weights, `max_shift` (default 3) and `max_stretch` (default 1.25).
Profiles are cached in `~/.cache/counterpoint/profiles.json`.

## Develop

```sh
uv run ruff check src tests
uv run pytest -q
```

CI runs on x86 and ARM, so the Raspberry Pi path stays installable from wheels.
