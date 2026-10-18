# Whisper test set

Measures how well different whisper.cpp models and settings transcribe this
frequency, against transcripts you write by hand.

It runs in Docker as its own Compose project (`whisper-testset`), separate from the recorder: its own image
with its own whisper.cpp build, its own models and data in this folder. It works the same whether the
recorder runs on bare metal or in containers. The only thing it reads from the recorder is the clip archive
(mounted read-only), and only when picking clips.

| File | What it is |
|---|---|
| `docker-compose.yml`, `Dockerfile` | The `transcribe` and `benchmark` services and their image |
| `transcribe.py` | Picks 20 clips from the archive and serves a page to replay and transcribe them |
| `benchmark.py` | Runs each configuration over the transcribed clips and scores it |
| `configs.json` | The models and settings to compare |

Run all commands below in this folder (`cd testset`).

Settings, in the environment or a `.env` file in this folder:

| Setting | Default | Purpose |
|---|---|---|
| `ARCHIVE` | `~/scanner-archive` | Recorder archive to pick clips from (bare metal). Use `../data/archive` once the recorder runs in Docker |
| `UID`, `GID` | `1000` | Owner of the files the containers write here. Check yours with `id -u` and `id -g` |
| `WITH_HF` | `1` | `1` includes the converter for `hf:` models (about 1 GB more image); `0` leaves it out |
| `WHISPER_THREADS` | all cores | CPU threads per transcription |

Created at runtime (ignored by git, like all recordings):

| Path | What it is |
|---|---|
| `data/testset.json` | One entry per clip: source clip, audio file, your transcript, notes, the live transcript |
| `data/clips/` | Copies of the selected clips (`NN_YYYY-MM-DD_HHMMSS-n.wav`), so the set survives `KEEP_DAYS` |
| `data/results/` | `summary.csv`, `report.html` and cached Whisper output per configuration |
| `models/` | ggml models downloaded or converted for the benchmark |

`transcribe.py` and `benchmark.py` use only the Python standard library; they also run without Docker
(`python3 transcribe.py --archive ~/scanner-archive`), but the benchmark needs the image's whisper.cpp.

## 1. Build the image

```bash
docker compose build
```

This compiles whisper.cpp (`whisper-cli` and `whisper-quantize`, same version as the timeline's Dockerfile) and,
with `WITH_HF=1`, installs CPU-only PyTorch and transformers for converting fine-tuned models. It takes a while
on the laptop. The build uses the host's network settings, so it works even where Docker's own DNS fails.

The build is optimised for the CPU it's built on, so build it on the machine that runs the benchmark.

## 2. Transcribe

```bash
docker compose up -d transcribe
```

Open http://adsb.local:8091. The first start picks 20 clips (1.5 to 30 s long, spread over the archived days)
and copies them into `data/`. Later starts keep the same set and don't need the archive. Stop it with
`docker compose down`.

- The clip plays when you open it. In the text field: Ctrl+Space play/pause, Esc replay from the start,
  Ctrl+[ / Ctrl+] skip 2 s, Ctrl+Enter save and go to the next clip, Ctrl+↑/↓ previous/next clip.
- Slower playback (0.6×, 0.75×) and Loop help with callsigns.
- Write numbers as digits or words (English or German: "1013", "one zero one three" and "eins null eins drei"
  all match) and callsigns as letters or NATO words; all are normalised before scoring. Write `??` for a word
  you can't make out; it isn't scored. Transcribe German traffic in German.
- **English / German** tags the clip's language, so the benchmark can score each language separately. Untagged
  clips are guessed from the transcript (German if it has at least two typical German words).
- **Only noise / no speech** saves an empty transcript. Those clips measure how often Whisper invents text.
- **Replace clip** swaps a clip that's unusable for another random one.
- What the live timeline produced is hidden under a fold, so it doesn't steer your transcript.

### Adding clips

Existing clips and transcripts are always kept; new clips are numbered after them.

**By hand (recommended for German clips):** **Browse archive** at the top of the page lists every archived
clip by day, with the live transcript. Click a row to play it, or use ↑/↓ to step through and play clips,
Space to pause and A to add the selected clip. "Play next automatically" plays the day back to back. The
filter box searches the live transcripts, but with `WHISPER_LANG=en` German speech often comes out as an
English translation or nonsense, so listening is more reliable than searching. **Remove clip** on the
Transcribe tab takes a clip (and its transcript) out of the set again.

**At random:** stop the running page first (`docker compose down`), then start it with options:

```bash
docker compose run --rm --service-ports transcribe --add 15 --lang de
```

| Option | Effect |
|---|---|
| `--add 15` | Adds 15 random clips to the set (`--count 30` instead tops it up to 30 in total) |
| `--lang de` | Adds only clips Whisper detects as German. Each candidate takes a few seconds to check with `whisper-cli --detect-language`; results are cached in `data/detected.json`. While the page runs with `--lang`, **Replace clip** also picks a German clip |
| `--detect-model base` | Model for the detection (downloaded into `models/` if missing). `tiny` is faster, `small` more reliable |
| `--min-prob 0.5` | Clips detected with lower confidence count as unsure and are skipped |
| `--min-duration` / `--max-duration` | Length filter in seconds (default 1.5 to 30) |
| `--seed 1` | Makes the random selection repeatable |

Whisper's guess is shown next to the language buttons as a hint; tag the clip yourself once you've heard it.
On this frequency the detection turned out unreliable (in a first try, 15 clips detected as German were all
English), so prefer picking German clips by hand. To start a fresh set, delete `data/`.

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
stored as `listen_filter` next to your transcript, so you can later tell which filters helped you. The filters
only change what you hear; the clip files are not modified.

## 3. Benchmark

```bash
docker compose run --rm benchmark
```

It gets missing models into `models/`, runs every configuration over every transcribed clip and
prints a ranking, best first. The columns:

- **WER / CER**: word / character error rate over all clips (lower is better). CER is kinder to near
  misses such as "Lufthansa" vs "Lufthansa's".
- **WER en / WER de**: the same per language, shown when the test set has both. Below the table, "Best
  per language" estimates what detecting each clip's language and then using the best model for it would
  score overall, compared with the best single model.
- **Sub / Del / Ins**: wrong, missing and invented words.
- **Noise→text**: clips you marked as no speech where Whisper still wrote something.
- **RTF**: transcription time divided by audio length, on the machine running the benchmark. On the
  recorder's laptop, above 1 means the timeline's backlog grows during busy traffic.
- `live timeline (stored)` scores the transcripts the timeline already saved, for free.

`data/results/report.html` shows every clip with each configuration's output and its errors
marked. While the transcribe service runs, it is also at http://adsb.local:8091/report.

Results are cached per configuration, so a rerun only transcribes what's new or changed. Options go after the
service name, e.g. `docker compose run --rm benchmark --only atc-small,small`: `--only`, `--list`, `--force`
(rerun everything), `--threads 2`, `--nice 19`, `--no-atc-normalize` (score on plain lower-cased words).

Whisper runs at low CPU priority (`nice 10`), so a benchmark on the recorder's laptop doesn't hold up
live transcription. The live timeline gets the CPU first, so speed figures measured while traffic is
busy come out slower than they really are. The larger models take a long time on a dual-core CPU; start
with `--only` on the quick ones.

To run it on a faster computer instead, copy the whole `testset/` folder there (with `data/`) and build the
image on that computer. Accuracy results carry over; speed results don't.

## Configurations

Each entry in `configs.json`:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Label in the results; also the cache file name |
| `model` | required | See below |
| `quantize` | none | Quantise the model first, e.g. `q5_0` or `q8_0`: smaller and faster, usually slightly less accurate |
| `lang` | `en` | `-l` value: `en`, `de` or `auto` |
| `prompt` | `true` | `true` = the timeline's ATC prompt, `"de"` = a German one with Flugfunk vocabulary, `false` = none, or a string for your own |
| `args` | `[]` | Extra `whisper-cli` arguments, e.g. `["-bs", "1"]` |
| `raw` | `false` | `true` skips the timeline's clean-up of noise tags and known hallucinations |

`model` can be:

| Form | Example | Source |
|---|---|---|
| whisper.cpp model name | `small`, `base.en`, `large-v3-turbo-q5_0` | Downloaded from whisper.cpp's Hugging Face repository |
| `hf:<owner>/<repo>` | `hf:jlvdoorn/whisper-small-atco2-asr` | A fine-tune in transformers format: downloaded and converted to ggml (needs the image built with `WITH_HF=1`) |
| URL of a `.bin` | `https://huggingface.co/<repo>/resolve/main/<file>.bin` | A ready-made ggml file |
| Path to a `.bin` | `/home/alex/ggml-mine.bin` | A local ggml file |

### ATC fine-tuned models

The standard Whisper models were trained on general internet audio. The `atc-` entries are community
fine-tunes trained on air traffic control recordings:

| Entry | Model | Trained on |
|---|---|---|
| `atc-base.en` | [jlvdoorn/whisper-base.en-atco2-asr](https://huggingface.co/jlvdoorn/whisper-base.en-atco2-asr) | ATCO2 (real European ATC radio) |
| `atc-small`, `atc-small no-prompt`, `atc-small q5_0` | [jlvdoorn/whisper-small-atco2-asr](https://huggingface.co/jlvdoorn/whisper-small-atco2-asr) | ATCO2 |
| `atc-small.en` | [jlvdoorn/whisper-small.en-atco2-asr](https://huggingface.co/jlvdoorn/whisper-small.en-atco2-asr) | ATCO2 |
| `atc-small atcosim` | [jlvdoorn/whisper-small-atcosim](https://huggingface.co/jlvdoorn/whisper-small-atcosim) | ATCOSIM (clean simulator recordings) |
| `atc-medium.en q5_0` | [jacktol/whisper-medium.en-fine-tuned-for-ATC](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC) | ATCO2 and UWB-ATCC; the author reports 15 % WER vs 95 % for the base model |

Error rates published for these models come from their own test sets and don't carry over to this frequency,
which is why this benchmark exists. Medium is probably too slow for live use on the laptop, but shows how
much a bigger ATC model would gain. More fine-tunes:
[huggingface.co/models?search=atc whisper](https://huggingface.co/models?search=atc%20whisper).

Converting a model takes a few minutes and downloads its full-precision weights (about 1 GB for Small, 3 GB
for Medium); only the converted ggml file is kept.

### German traffic

There is no public Whisper fine-tune for German radio traffic. The ATC models above were trained on English
transcripts; ATCO2 and ATCOSIM have many German-accented speakers, but their German skills may have suffered
from the fine-tuning. The German entries compare the options:

| Entry | What it tests |
|---|---|
| `atc-small de`, `atc-small auto` | How much German the English ATC fine-tune still handles |
| `small de`, `medium-q5_0 de` | Standard multilingual models with German set and a German Flugfunk prompt |
| `small auto`, `small auto prompt-de` | Automatic language detection, as mixed traffic would need |

`lang: de` forces German, which helps German clips and hurts English ones. That's why the per-language
columns and the "Best per language" line matter: if a German configuration wins on German clips and
`atc-small` on English ones, detecting the language per clip and choosing the model accordingly is worth
building into the timeline. With only a few German clips in the set, the German figures aren't reliable;
add German clips with **Browse archive** (see "Adding clips").

## Using the winner

The benchmark runs `whisper-cli` the same way the timeline does (`-nt -np`, same prompt and clean-up), so a
winning model file from `testset/models/` works in the timeline as is. Copy it to the repository's top-level
`models/` folder and set `WHISPER_MODEL=/extra-models/<file>` in the top-level `.env`, e.g.:

```bash
cp testset/models/ggml-jlvdoorn--whisper-small-atco2-asr.bin models/
```

Then `docker compose up -d` in the repository folder. With `RETRANSCRIBE=1` (the default) the timeline also
redoes the existing archive with the new model in the background. Settings other than the model, language and
prompt need a change in `timeline/server.py`.
