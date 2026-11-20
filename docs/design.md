# Counterpoint: design

Counterpoint is the harmonic sample matcher (renamed 2026-11-19).

Status: designed, not built. Revised 2026-11-19.

> This is a reconstruction. The original `harmonic-matcher/design.md` lives in the old
> Sound project's files and didn't come across. This version is rebuilt from the Sound
> handover summary plus How's rule from 2026-11-19 (winnower never changes samples).
> If the original turns up, reconcile any detail it has that this one lacks.

## What it does

Given a chord (or a passage) you're playing, find samples in winnower's catalogue that sit
well against it, say how far each one would need to be shifted and stretched to fit (in
pitch and in tempo), and suggest a small set that layer well together.

It is not a nearest-neighbour search over embeddings. It scores each sample's 12-note pitch
profile directly against the chord, at all 12 transpositions, so the reasons for a match
are explicit (fits the scale, adds colour, isn't rough) and tunable.

It has three modes: `match` ranks samples against a chord you give it, `pack` builds a
key-sorted folder of samples that combine well, for Ableton, and `voice` runs live as one
of several voices that listen to each other and change sample every few loops (see
Ensemble mode).

## Boundary with winnower

Winnower collects and catalogues. It never modifies or transposes samples, because the
same sample gets reused in many contexts. So:

- Counterpoint is a **separate tool** (its own package, its own CLI). It is not a
  winnower subcommand and adds nothing to winnower's code.
- It **reads** winnower's storage (local path now, S3 later) through the same layout:
  `index/samples.jsonl` for the list, `samples/**/*.json` sidecars for detail, the FLAC
  files for audio. It never writes into winnower's storage.
- Transposition and tempo matching are **suggestions**, reported per match as a shift in
  semitones and a stretch to a target tempo. Stored audio is never pitch-shifted,
  time-stretched, re-rendered or re-saved. The person (or Ableton) applies them at use
  time, so the same sample can be used at different shifts and tempos elsewhere.
- Anything Counterpoint computes (pitch profiles, caches, match results) lives in the
  Counterpoint's own store, keyed by winnower's sample `id` plus a hash of the FLAC, so a
  re-harvested or re-cut sample invalidates its cache entry.

Winnower's sidecars carry key, Camelot code, BPM and bars, but not a 12-bin pitch profile,
so Counterpoint computes that itself. Winnower's key and Camelot fields are used only as a
cheap pre-filter.

## Inputs

The chord can come from:

1. **Chord text**: `Am9`, `Fmaj7#11`, `C/G`. Parsed into weighted pitch classes (root
   and third weighted higher than extensions).
2. **MIDI**: a `.mid` file (a bar or a time range of it) or live MIDI in. Notes are
   folded to pitch classes, weighted by duration and velocity.
3. **Audio**: a clip of what you're playing. Transcribed to notes with basic-pitch, then
   treated like MIDI. A chroma fallback (librosa `chroma_cqt`) is used for dense or
   noisy audio where transcription is unreliable.

The **tempo** of the input is taken alongside the chord:

- MIDI: the file's tempo map, or the clock on live MIDI in.
- Audio: librosa beat tracking, the same way winnower detects tempo (whole-recording
  estimate, checked in windows). If no steady pulse is found, the input is treated as
  free time and tempo matching is skipped.
- Chord text: `--bpm N`, or none.
- `--bpm N` always overrides a detected tempo.

Optionally a **key or scale** for context (`--scale "A dorian"`). Without one, the scale
is inferred from the chord: chord tones plus the most likely diatonic completion.

## Sample pitch profile

For each catalogued sample:

- `chroma_cqt` over the loop, averaged, normalised to sum to 1: the 12-bin profile `p`.
- A **pitchedness** measure (contrast of the chroma, as winnower's `tonal.py` already
  uses with `MIN_CONTRAST`). Samples below the floor are **unpitched**: noise, crackle,
  rain, most percussion. They're never scored for harmony; they're offered separately as
  texture layers with shift 0.
- Spectral centroid and band energy (low/mid/high), reused from the sidecar where present,
  for masking checks when layering.

Computed once per sample and cached. A Pi 5 can do this incrementally after each harvest.

## Scoring a sample at a transposition

For chord weights `h` (12 bins), scale mask `s`, sample profile `p` and shift `t` in
-6..+5 semitones, let `q = roll(p, t)`.

- **Scale fit**: share of `q`'s energy on scale tones. Energy outside the scale is the
  main thing that makes a layer sound wrong.
- **Added colour**: share of `q`'s energy on scale tones that aren't chord tones (9ths,
  11ths, 13ths). Rewarded up to a point: a little colour is the point of layering, a lot
  of it reharmonises the chord.
- **Roughness**: a pairwise dissonance sum between `q` and `h`, weighting each interval
  class (minor second and tritone high, fifths and octaves near zero), as in a simple
  Plomp-Levelt style table on pitch classes. Only `q`'s energy off the chord tones
  counts, because a sample that doubles the chord adds no roughness the chord doesn't
  already have. Penalised.
- **Shift cost**: a small penalty growing with `|t|`, because big shifts change a
  texture's character (grain, formants, speed if repitched by resampling).

`score = w_fit * fit + w_col * min(colour, colour_cap) - w_rough * roughness - w_shift * |t|`

All 12 transpositions are scored for every sample; the best `t` within the allowed range
is reported. Weights live in a config file and are expected to be tuned by ear.

## Picking a layering set

From the top-scoring pitched samples, choose a set of N (default 3) greedily:

1. Take the best single match.
2. Add the candidate that maximises the score of the **combined** profile (sum of the
   chosen, shifted profiles) against the chord, so the set as a whole fits, not just each
   member.
3. Penalise candidates that pile energy into the same band as samples already chosen
   (masking), and near-duplicates from the same source recording.
4. Optionally add one unpitched texture layer.

## Matching tempo

When the input has a tempo, each sample also gets a suggested **stretch**. Winnower cuts
every loop to 1, 2, 4 or 8 bars at a whole-number BPM and records it in the sidecar
(`bpm`, `bars`, `tempo_source`), so the stretch is a ratio and the loop stays a whole
number of bars at the new tempo.

- Ratio = target BPM / sample BPM, trying the sample at half, normal and double time and
  keeping whichever ratio is closest to 1. A 72 BPM loop against a 140 BPM input plays at
  double time with a ratio of 0.97, not 1.94.
- **Stretch cost**: a penalty growing with `|log2(ratio)|`, added to the score alongside
  the shift cost. It's steeper for `tempo_source: detected` samples (a real groove smears
  audibly when stretched) and gentler for `fitted` ones (rain, drones, crackle, whose
  "tempo" is only a loop length).
- `--max-stretch R` limits the ratio (proposed default 1.25, i.e. up to 25% either way);
  samples outside it are dropped rather than ranked low. `--no-stretch` falls back to
  using the BPM only as a filter.
- Pitch and time are suggested independently, assuming the DAW stretches without
  changing pitch (Ableton's Complex/Complex Pro or Texture warp modes). If someone
  repitches by resampling instead, the speed change moves the pitch too, so the report
  also gives the combined resample ratio for that case.

With no input tempo, stretch is skipped and `--bpm` acts as an optional filter on
winnower's `bpm` field, as before.

## Ensemble mode

Counterpoint can also run live, as several **voices** that listen to each other. Each
instance is one voice: it plays one sample on a loop, and after a set number of loops it
picks a new sample that fits with what the other voices are playing at that moment. It's
a slow, changing layering set, where the choice is spread across the voices and made
again whenever one of them changes.

### Starting a voice

```sh
counterpoint voice --loops 4-8            # change after 4 to 8 loops, chosen at random each time
counterpoint voice --loops 6 --anchor Am9 # change after exactly 6 loops; stay near A minor 9
counterpoint voice --name low --band low  # prefer samples with most of their energy in the low band
```

Voices can run in separate processes on one machine, or one per Raspberry Pi on the same
network. They find each other automatically, with nothing to configure. The laptop is
the first target and Pis the end goal. See Spatial installations on Raspberry Pis.

### Shared clock

Voices share tempo and bar position through **Ableton Link**, the open protocol Ableton
uses to keep devices in time on a local network. Every voice joins the same Link session,
so loops start on the same downbeats, and Ableton, if it's open, can join the session to
play or record in time with them. The ensemble tempo is the Link tempo: set by the first
voice (`--bpm`), by Ableton, or by a tempo detected from an input (see below). That target
tempo drives the stretch matching above.

### Listening

"Listening" is done over the network, not with microphones. Each voice sends a short
message (OSC over UDP multicast) whenever its state changes, and every voice keeps track
of the others' state:

- its name, and the sample it's playing now: id, suggested shift, stretch ratio
- that sample's 12-bin pitch profile after the shift, and its band energy
- when it will next change (the bar number on the Link timeline), and what it plans to
  play next once it has chosen

Sending what each voice plays is exact and cheap: no pitch detection on mixed audio, and
it still works when the voices play out of the same speakers. An optional **input** (a
microphone, a line in, or MIDI from a person playing) is analysed as described under
Inputs and treated as another voice that never changes sample. That's how the ensemble
follows a performer.

### Choosing the next sample

One loop before its change, a voice scores candidates using the same scoring as above,
with these differences:

- The reference isn't a typed chord. It's the **combined profile of the other voices**,
  using each voice's planned next sample where one has been announced, plus the anchor
  and the input when they're present.
- **Anchor**: `--anchor Am9` (or a key) adds a fixed reference with a weight, so the
  harmony can drift but is pulled back. Without an anchor and input, the harmony
  wanders freely. Each choice only has to fit the voices playing at that moment, so over
  many changes the ensemble can modulate. That's deliberate, and the anchor weight sets
  how much.
- **Novelty**: samples this voice played recently, and any sample another voice is
  playing, are excluded. A candidate from the same source recording as another voice's
  sample is penalised.
- **Band**: masking against the other voices is penalised, and `--band` lets a voice
  favour low, mid or high so each voice keeps a role.
- **Unpitched voices**: `--texture` makes a voice pick only from unpitched samples
  (noise, rain, crackle). Its choice is based on tempo, band and novelty only.

A standalone voice pitch-shifts and time-stretches the chosen sample **in memory** during
the last loop, and the new buffer fades in on the bar where the change falls. The
previous sample has already faded out, by the bar line after its last loop, and the
voice has rested at least a bar in between (a voice playing alone crossfades straight
into its next sample instead). An Ableton voice hands the shift and stretch to Ableton instead (see Sound
output). Either way nothing is written to disk, and winnower's files are never changed.

### Avoiding clashes

No two voices start a sample on the same bar, so entries are staggered and each one is
heard. A voice claims its next start bar as soon as it knows when its current sample
ends, skipping bars other voices have already claimed. If two voices claim the same bar
at once, the one that has already locked it (it is inside its preparation window) keeps
it; otherwise the earlier name does, and the other moves on to the next free bar.

Each choice takes the other voices' announced next picks into account. As a fallback,
voices that still end up starting on the same bar choose one at a time, in name order,
each announcing its pick before the next chooses. If a voice stops sending for a few
bars, the others drop it.

### Sound output

A voice has two ways to make sound. Both kinds of voice can be in the same ensemble,
because they share the Link clock and the OSC messages.

**Standalone (default).** The voice plays through its own audio output with
sounddevice/PortAudio, rendering each pick in memory as described above. No Ableton is
needed, so a set of Pis, one per speaker, can run an installation unattended. `--device`
selects an output, so several voices on one machine can use separate channels of an
audio interface.

**Ableton (`--output ableton --track NAME`).** The voice still listens and chooses the same
way, but Ableton plays the sound. A **Max for Live device** on the named audio track
receives each pick over OSC and, through the Live API:

1. loads the untouched FLAC into a free clip slot on that track (Live 11 and later read
   FLAC),
2. sets the clip's Transpose to the suggested shift, turns Warp on, and sets the clip's
   original tempo and loop length from winnower's `bpm` and `bars`,
3. launches the clip quantised to the change bar, and clears the old clip after it stops.

Ableton then does the shifting and stretching, so the voice skips its own rendering, and
you get Live's effects, mixing and recording on each voice. The FLACs have to be on a disk
the Ableton machine can read, such as a synced copy of winnower's storage. The Live API
call that creates an audio clip from a file path is in Live 12. Before building on it,
check that it can be called from Max for Live on your Live version.

**Selecting samples in Ableton by hand.** The same device has a **Suggest** mode for when
you're arranging rather than running an ensemble. It reads the clips playing on the other
tracks (or a chord you type), asks Counterpoint for matches, and shows a ranked list with
each sample's shift and stretch. Clicking one loads it onto the device's track with
Transpose and Warp set, as above. This is `match` mode with Ableton as the interface.
Counterpoint runs as a small local service that the device talks to over OSC.

### Logging

Each voice writes a log of what it played: sample id, shift, stretch, the bar it started
and how long it played. The log can be replayed, and its contents fed back into winnower
ratings later.

## Spatial installations on Raspberry Pis

The goal is one voice per Raspberry Pi, each with its own speaker, on one network. The
devices are autonomous: each one chooses its next clip by listening to the others, and
there's no central controller. The plan is to get it working on the laptop first, using
the same code and the same messages, so moving to Pis only changes where things run.

### Stage 1: laptop

- `counterpoint ensemble --voices 4` starts four voice processes on the laptop. Each is
  an ordinary `counterpoint voice`, the same as it will be on a Pi, with its own name,
  config, Link peer and OSC traffic. The launcher only saves opening four terminals.
- Output: each voice goes to its own channel pair on an audio interface (`--device`,
  `--channels`), or all go to the laptop's speakers with `--pan` to tell them apart.
- Library: voices read from a pack folder, the same way a Pi will. That means the
  laptop stage exercises the path the Pis use, not a shortcut through winnower's storage.
- Ableton voices and the Suggest device are laptop-only. They can join the same
  ensemble as the standalone voices.

### Stage 2: Pis

Each Pi runs one `counterpoint voice` as a service and needs nothing else at show time.

**Self-contained library.** Each Pi holds its own pack on its SD card or SSD, synced
before the show. Packs carry each sample's cached pitch profile, band energy, BPM and
bars in `pack.json`, so a Pi never analyses audio at run time and never needs winnower's
storage. Pis can share one pack, or each can get its own: a bass pack for the low
voices, textures for others.

**Keeps playing if the network drops.** If a voice hears no one, it keeps playing and
choosing, using the last state it heard plus the anchor, and its own Link clock carries
on. When the network returns, Link resyncs to the session and the voices pick each other
up. A lost peer never stops a voice.

**Network.** Link and the voices' messages use multicast, which many Wi-Fi access
points throttle or block, and client isolation on venue Wi-Fi stops it entirely. The
recommended setup is a dedicated router for the installation, with Ethernet to each Pi
where cables are practical. As a fallback, each voice can list its peers' addresses in
its config and send to each one directly (unicast). Link has no unicast fallback, so on a
network that blocks multicast the voices keep their tempo with a slower drift correction
over the OSC channel instead.

**Audio hardware.** The Pi 5 has no headphone jack. Each Pi needs a USB audio interface
or a DAC HAT (for example a HiFiBerry), feeding an active speaker.

**Unattended running.**

- A systemd service starts the voice at boot and restarts it if it crashes.
- Each voice's settings live in a per-device `counterpoint.toml`: name (defaults to the
  hostname), band, loops, anchor, output device, volume and pack path.
- A read-only root filesystem (Raspberry Pi OS overlay) survives venue power cuts. Logs
  go to a small writable partition, or are sent over the network.
- A health check on each device marks it unready if its audio device or pack is
  missing.

**Running the room from one place.** `counterpoint ensemble status` lists every voice it
can hear: what each is playing, its key after shift, and when it next changes.
`counterpoint ensemble set anchor Dm9` (or tempo, or volume) broadcasts a change that
every voice applies at its next change, so a performer or a timer can steer the piece
without touching any Pi. Nothing in the ensemble depends on this command running.

**Space (optional).** A voice can have a position in its config (`position = [x, y]` in
metres). When choosing, it can weight closer voices more heavily, so neighbouring
speakers agree more than speakers across the room do. The harmony then varies as you
walk through the space. It's off by default and best tried once the basics work.

**Pi 5 load.** Choosing is cheap: scoring a few hundred cached profiles at 12 shifts is
milliseconds. The cost is rendering the next sample with Rubber Band (pitch and stretch)
during one loop of the current one. That should be comfortable for a stereo stream on a
Pi 5, but it's an estimate until benchmarked. If it's tight, voices can render at lower
quality settings, or packs can be built with `--bpm` so most samples need no stretch.

## Packs

`counterpoint pack` picks a set of samples that combine well and copies them into a
folder, sorted by key, for use in Ableton and as the library each Pi voice plays from. It runs ahead of time, so Ableton never needs a
connection to winnower's storage.

```sh
counterpoint pack --count 100 --out ~/Music/Counterpoint/packs          # mutually compatible set
counterpoint pack --count 100 --anchor Am9 --bpm 96 --out ...           # fitted to a chord and tempo
counterpoint pack --count 100 --texture 20 --commercial --out ...       # 20 of them unpitched; release-safe licences only
```

### Choosing 100

It's the layering-set selection scaled up. Picking 100 that all fit one chord would give
100 near-identical samples, so a pack aims at a **harmonic neighbourhood** instead:

- With `--anchor`, the centre is that chord. Without one, Counterpoint chooses the centre
  key that has the most strong candidates in the catalogue, and says which it chose.
- Samples are chosen greedily, like the layering set. Each one has to fit the anchor and
  must not be rough against the pack's combined profile. Its key after the suggested
  shift must be the centre key or a neighbour on the Camelot wheel, so any two samples in
  the pack can be layered after at most a small Transpose.
- Diversity is enforced across the pack: a spread of bands, of `bars` lengths, and of
  source recordings (at most a few per recording), plus `--texture N` unpitched samples
  that fit any key.
- With `--bpm`, the stretch limit applies as usual. Without it, samples of any tempo are
  allowed, and set.json records each one's BPM so Ableton's warp can bring them together.

### Folder layout

```
packs/2026-11-19 Am9 100/
  8A A minor/   +2_Am_96bpm_4bar_10.0s_pulsing-dark-samba-de-fato_29424a.flac
  8B C major/   0_C_110bpm_2bar_4.4s_drifting-bright-organ_77b1e0.flac
  9A E minor/   ...
  no key/       0_nokey_108bpm_4bar_8.9s_crackling-airy-rain-on-tin-roof_8f31c0.flac
  pack.json     every sample: id, source, licence, attribution, original key, shift,
                key after shift, bpm, bars, scores, cached pitch profile and band
                energy (so a Pi voice never re-analyses)
  ATTRIBUTION.txt
```

- Files are **copies** of winnower's FLACs, byte for byte, so the pack works on any machine
  and winnower's storage is untouched. Nothing is transposed or stretched.
- Folders are named by the key each sample **sounds in after its suggested shift**,
  because that's how you'll use them in Ableton. The shift is the first part of the file
  name (`+2_...`): set the clip's Transpose to it. The original key stays in the name and
  in `pack.json`.
- `ATTRIBUTION.txt` collects the attribution text for every CC BY sample, ready for
  release notes. Any flagged licence (NC, SA, unlicensed) gets a warning at the top.

### Getting it to the Ableton machine

Counterpoint writes the pack where it runs (Pi or server). A sync tool then copies it to
the Mac, either Syncthing (continuous, recommended) or `rsync` after the run. To stop a
half-copied pack from looking finished, Counterpoint builds it in a hidden `.building-...`
folder, renames it when complete, and writes `READY` inside last.

### Saying it's ready

When the pack is complete, Counterpoint prints a summary: the centre key, how many
samples went into each key folder, and any licence warnings. It can also send a
notification with `--notify URL`, a plain HTTP POST that works with ntfy (phone push) or
any webhook. It reports progress as it goes, since profiling a large catalogue on a Pi
takes a while the first time. After that, cached profiles make packs fast.

The Ableton device's Suggest mode and Ableton voices can use a synced pack as their
library, so the Ableton machine never needs to reach winnower's storage directly.

## Output (match mode)

`counterpoint match` reports; it doesn't render. Ensemble mode is the part that plays.

- A table (and `--json`) per match: sample id, file path, winnower key/Camelot, suggested
  shift in semitones, suggested target tempo and stretch ratio (with half/double time
  noted), score and its parts (fit, colour, roughness, shift and stretch cost), licence
  decision.
- `--export DIR` writes a **set folder** for Ableton: symlinks (or copies where the
  filesystem can't link) to the untouched FLACs, with the shift in the link name, e.g.
  `+3_Am_96bpm_4bar_...flac`, plus `set.json`. In Ableton, the shift goes straight into
  the clip's Transpose control, so the stored file stays as it is. `set.json` also
  records the target tempo and each sample's original BPM and bar count, so warping in
  Ableton (set the clip's original tempo, warp on, project at the target tempo) lands each
  loop on whole bars. Writing an Ableton `.asd` warp file per sample would make that
  automatic, but `.asd` is undocumented, so it's a later maybe.
- Licence decision is carried through so `--commercial` behaves as it does in
  `winnower list`.

## Resolved and reframed questions

**Play from the tool or send to Ableton?** Both, by mode. `match` ranks and suggests,
and hands Ableton untouched files plus a shift and stretch to apply. `voice` (ensemble
mode) plays itself by default, shifting and stretching in memory, so it runs without
Ableton. It can also play through Ableton tracks via a Max for Live device, which also
offers `match` as a Suggest list inside Live. In no mode is any stored audio changed.

**Transposition on by default?** Resolved. Stored audio is never transposed, so there's
nothing to turn on. Scoring at all transpositions is always on, because that's how the
match is found. What's configurable is the **allowed range** of suggested shifts:
`--max-shift N` (proposed default 3 semitones either way, to keep textures recognisable)
and `--no-shift` to only accept samples that fit as they are.

## Platform

- Raspberry Pi 5 (64-bit OS) or a small x86 server, same as winnower.
- librosa for chroma; basic-pitch for audio-to-notes; mido for MIDI. basic-pitch's ARM
  support (TFLite/ONNX runtime) needs checking on the Pi before relying on audio input;
  chord text and MIDI input don't need it.
- Python, managed with uv, matching winnower.
- Ensemble mode: Ableton Link through a Python binding (aalink or LinkPython; Link
  itself is GPLv2), python-osc for messages, sounddevice for audio out, and Rubber Band
  (via pyrubberband or pedalboard, both GPL) for shifting and stretching. Rendering a
  4 to 8 bar loop with Rubber Band during one loop of playback should be easy for a Pi 5,
  but this is an estimate. Benchmark it before relying on it.
- Ableton integration: a Max for Live device (Live Suite, or Standard with the Max for
  Live add-on), written in Max with a JS (or Node for Max) part for the Live API calls
  and OSC.

## Not in scope

- Modifying, transposing or re-saving samples (winnower's rule, and this tool's too).
- Embedding or similarity search (winnower's planned CLAP work could feed a separate
  "sounds like" filter later).

## Open questions

- Where the code lives: a new repo (e.g. `melnyczuk/counterpoint`) fits the
  separation best.
- Which Live version and edition you run, which decides whether the device can load a
  clip from a file path directly.
- Pi audio hardware (USB interface or DAC HAT), and whether the installation gets its own
  router and Ethernet.
- Pack defaults: how many samples per source recording, how wide the Camelot
  neighbourhood is (one step proposed), and where on the Mac packs land.
- Ensemble defaults: loop count range, anchor weight, and how quickly a silent voice is
  dropped.
- Default weights, the shift range and the stretch limit: proposals above, to be tuned
  by ear on real winnower output.
