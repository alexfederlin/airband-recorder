# Whisper test set

Measures how well different whisper.cpp models and settings transcribe this
frequency, against transcripts you write by hand.

| File | What it is |
|---|---|
| `transcribe.py` | Picks 20 clips from the archive and serves a page to replay and transcribe them |
| `benchmark.py` | Runs each configuration over the transcribed clips and scores it |
| `configs.json` | The models and settings to compare |

Created at runtime (ignored by git, like all recordings):

| Path | What it is |
|---|---|
| `data/testset.json` | One entry per clip: source clip, audio file, your transcript, notes, the live transcript |
| `data/clips/` | Copies of the selected clips (`NN_YYYY-MM-DD_HHMMSS-n.wav`), so the set survives `KEEP_DAYS` |
| `data/results/` | `summary.csv`, `report.html` and cached Whisper output per configuration |
| `models/` | Models downloaded for the benchmark |

Both scripts use only the Python standard library.

## 1. Transcribe

Run it inside the timeline image, which already has the archive mounted (from the repository folder):

```bash
docker compose run --rm -p 8091:8091 -v ./testset:/testset timeline python3 /testset/transcribe.py
```

Open http://adsb.local:8091. The first start picks 20 clips (1.5 to 30 s long, spread over the archived
days) and copies them into `testset/data/`. Later starts keep the same set.

- The clip plays when you open it. In the text field: Ctrl+Space play/pause, Esc replay from the start,
  Ctrl+[ / Ctrl+] skip 2 s, Ctrl+Enter save and go to the next clip, Ctrl+↑/↓ previous/next clip.
- Slower playback (0.6×, 0.75×) and Loop help with callsigns.
- Write numbers as digits or words and callsigns as letters or NATO words; both are normalised before
  scoring. Write `??` for a word you can't make out; it isn't scored.
- **Only noise / no speech** saves an empty transcript. Those clips measure how often Whisper invents text.
- **Replace clip** swaps a clip that's unusable for another random one.
- What the live timeline produced is hidden under a fold, so it doesn't steer your transcript.

### Listening filters

The **Listening filters** panel cleans up the audio in the browser while you listen, to check whether
filtering makes transmissions easier to understand. Ctrl+B (or the **Filters on/off** button) switches between
filtered and original audio mid-clip for A/B comparison. Settings are remembered between clips.

| Control | What it does |
|---|---|
| High-pass | Cuts hum and rumble below the set frequency (speech starts around 300 Hz) |
| Low-pass | Cuts hiss above the set frequency (radio speech ends around 3400 Hz) |
| Noise reduction | Learns the steady background noise from the quietest parts of the clip and subtracts it. High settings sound "watery" |
| Volume | Extra gain, mainly to make up for the leveller |
| Even out loud and quiet speakers | Compressor, so a weak pilot is as loud as the controller |

Presets: **Voice band** (300 to 3400 Hz), **+ level** (adds the compressor and +6 dB), **+ denoise** (adds noise
reduction at 50 %).

Under the controls the page shows roughly the same chain as an ffmpeg `-af` filter (ffmpeg's `afftdn` and
`acompressor` aren't identical to the browser's algorithms). When you save a clip with filters on, that string is
stored as `listen_filter` next to your transcript, so you can later tell which filters helped you and try them on
Whisper. The filters only change what you hear; the clip files are not modified.

Options: `--count 30` adds clips up to 30, `--min-duration` / `--max-duration` change the length filter,
`--seed 1` makes the selection repeatable. To start a fresh set, delete `testset/data/`.

## 2. Benchmark

```bash
docker compose run --rm -v ./testset:/testset timeline python3 /testset/benchmark.py
```

It downloads missing models into `testset/models/`, runs every configuration over every transcribed
clip and prints a ranking, best first. The columns:

- **WER / CER**: word / character error rate over all clips (lower is better). CER is kinder to near
  misses such as "Lufthansa" vs "Lufthansa's".
- **Sub / Del / Ins**: wrong, missing and invented words.
- **Noise→text**: clips you marked as no speech where Whisper still wrote something.
- **RTF**: transcription time divided by audio length. Above 1, the timeline's backlog grows during busy
  traffic.
- `live timeline (stored)` scores the transcripts the timeline already saved, for free.

`testset/data/results/report.html` shows every clip with each configuration's output and its errors
marked. While `transcribe.py` runs, it is also at http://adsb.local:8091/report.

Results are cached per configuration, so a rerun only transcribes what's new or changed. Other options:
`--only base,small`, `--list`, `--force` (rerun everything), `--threads 4`,
`--no-atc-normalize` (score on plain lower-cased words).

## Configurations

Each entry in `configs.json`:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Label in the results; also the cache file name |
| `model` | required | whisper.cpp model name (`tiny`, `base.en`, `small-q5_1`, `large-v3-turbo-q5_0`, …) or a path to a `.bin` |
| `lang` | `en` | `-l` value: `en`, `de` or `auto` |
| `prompt` | `true` | `true` = the timeline's ATC prompt, `false` = none, or a string for your own |
| `args` | `[]` | Extra `whisper-cli` arguments, e.g. `["-bs", "1"]` |
| `raw` | `false` | `true` skips the timeline's clean-up of noise tags and known hallucinations |

The benchmark runs `whisper-cli` the same way the timeline does (`-nt -np`, same prompt and clean-up), so
the winner can be used as is: set `WHISPER_MODEL_NAME` in `.env` and run `docker compose up -d --build`.
Settings other than the model and language need a change in `timeline/server.py`.

The large models are slow on the laptop. Run them with `--only` once the quick ones are done.

## Without Docker

```bash
cd testset
python3 transcribe.py --archive ../data/archive
python3 benchmark.py --whisper-bin ~/whisper.cpp/build/bin/whisper-cli
```

Files written by the container are owned by root; use `sudo` to delete `testset/data/` by hand.
