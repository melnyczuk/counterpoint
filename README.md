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
command (a key-sorted folder of samples for Ableton), the ensemble (below), and running
one voice per Raspberry Pi in a spatial installation. Tasks are tracked as GitHub issues.

## Status (v0.1)

- [x] Read-only winnower catalogue (local path or `s3://` with the `s3` extra)
- [x] Pitch profiles (chroma, pitchedness, band shares), cached outside winnower's storage
- [x] Targets from chord symbols, MIDI files and audio clips, with inferred or given scale
- [x] Scoring at all 12 transpositions: fit, colour, roughness, shift cost
- [x] Tempo matching: stretch ratio with half/double/quad time, slowing preferred to speeding
- [x] Layering sets: combined fit, band masking and same-recording penalties, texture layer
- [x] `counterpoint match` with table, JSON and an Ableton set-folder export
- [x] Ensemble: `voice`, `ensemble` (start, status, set), `listen`, `replay`
- [ ] `counterpoint pack`
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
- **stretch**: playback speed to reach the target tempo. `half`, `double` or `quad` means
  it's counted at half, double or quad time, so it's warped as such. Slowing a sample down
  is preferred to speeding it up: a ratio below 1 can go to 0.25 (four times as slow),
  but above 1 only to 1.25.
- `*` after a file name means its licence is flagged (NC, SA or unknown): clear it before
  release.

Chord symbols: `C`, `Am`, `F#m7`, `Bbmaj7`, `Fmaj7#11`, `G7b9`, `Bø`, `Bm7b5`, `Cdim7`,
`Caug`, `Dsus4`, `Esus2`, `Cadd9`, `C6/9`, `E5`, `G7alt`, `C/G`.

## Ensemble

Several voices, each looping one sample and changing it every few loops to fit what the
others are playing. Voices share tempo and bar lines through Ableton Link, and tell each
other what they're playing over OSC multicast on the local network. Run them on one
laptop now, and later one per Raspberry Pi.

```sh
counterpoint ensemble --voices 4 --anchor Am9 --bpm 90      # four voices, panned across stereo
counterpoint ensemble --voices 4 --spread-channels          # voice N on outputs 2N-1,2N
counterpoint ensemble --voices 5 --textures 1               # one of them plays only textures
counterpoint voice --name low --band low --loops 6          # a single voice (one per Pi)
counterpoint ensemble status                                # who's playing what, next change
counterpoint ensemble set anchor Dm9                        # every voice, from its next pick
counterpoint ensemble set volume 0.5 --voice v2
counterpoint ensemble --voices 4 --swell-bars 3,4,5,7     # each voice swells on its own period
counterpoint ensemble set swell 0                           # hold every voice's level steady
counterpoint ensemble --voices 4 --rest-bars 2 --fade-ms 400  # longer rests and fades
counterpoint listen --midi                                  # a keyboard player joins as a voice
counterpoint replay ~/.local/state/counterpoint/*.jsonl --out run.wav
```

- **Picking:** two bars before its change (`--prepare-bars`), a voice scores the library
  against the other voices' combined profile (their announced next picks where there
  are any), plus the input and `--anchor`. It draws from the best few and announces its
  pick.
- **Playing:** the pick is shifted and stretched in memory with Rubber Band, so the loop
  lasts a whole number of beats at the session tempo. It fades in on its bar over
  `--fade-ms 200`. Nothing is written to disk.
- **Levels:** every loop is brought to -20 LUFS (BS.1770 loudness, ignoring silences)
  before it plays, so a hot sample doesn't bury the others. Quiet ones are boosted by at
  most 12 dB, and peaks stay under 0.9. `--no-normalise` plays loops at their own level.
- **Swells:** each voice's level rises and falls slowly, `--swell 6` dB peak to trough,
  over about `--swell-bars 4` bars. The period drifts by up to `--swell-drift 0.15` every
  loop, so the voices phase against each other and each one comes to the fore now and
  then. `ensemble` gives each voice a slightly different period, or takes a list. Change
  them live with `set swell|swell-bars|swell-drift VALUE [--voice vN]`.
- **Loops:** `--loops 4-8` draws a new count each time. The sample fades out so it's
  silent by the next bar line after that many loops.
- **Rests and staggered starts:** with other voices playing, a voice rests at least
  `--rest-bars 1` bars before its next sample, and no two voices start a sample on the
  same bar: a voice claims its start bar when it schedules it, and moves to the next free
  one if another voice got there first. A voice on its own doesn't rest; its next sample
  crossfades in. Change them live with `set rest|fade VALUE [--voice vN]`.
- **Sync:** Ableton (Live 11+) joins the same Link session if Link is on in its
  preferences. Use `--no-link` for local time, and `--peers host:port` where the network
  blocks multicast.
- **Logs:** each voice logs its picks to `~/.local/state/counterpoint/NAME.jsonl`.
  `replay` renders a stereo mix of a run from those logs.
- **No sound card:** `--dry-run` schedules everything silently, which is useful for
  checking a setup.

On Linux (and the Pi), install PortAudio for audio out: `sudo apt install libportaudio2`.
For now, voices read straight from the winnower catalogue. Packs (#11) will replace that
for the Pis.

**Licences:** Rubber Band (through pedalboard) is GPLv3 and Link is GPLv2+, so builds
of Counterpoint that include them are distributed under the GPL.

## Settings

Everything is optional. See `src/counterpoint/config.py` for the full `counterpoint.toml`,
including the scoring weights, `max_shift` (default 3), `max_stretch` (fastest playback,
default 1.25) and `max_slow` (slowest, as a factor, default 4).
Profiles are cached in `~/.cache/counterpoint/profiles.json`.

## Develop

```sh
uv run ruff check src tests
uv run pytest -q
```

CI runs on x86 and ARM, so the Raspberry Pi path stays installable from wheels.
