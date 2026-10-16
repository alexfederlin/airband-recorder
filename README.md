# Airband recorder

Records an airport frequency from a scanner, splits it into one clip per transmission, and makes the recordings searchable in the browser.

- **Recorder** (`recorder/`): captures the scanner audio from a USB sound card, cuts it into clips, archives each clip and uploads it to [Rdio Scanner](https://github.com/chuot/rdio-scanner).
- **Timeline** (`timeline/`): a small Python server (standard library only) that transcribes archived clips with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and serves a day-by-day timeline with search and continuous playback.
- **Services** (`systemd/`): unit files that start everything at boot.

Built for a ThinkPad Yoga 12 running Xubuntu (hostname `adsb`, user `alex`), a Quansheng UV-K5(8) and a C-Media USB sound card. Adjust user names and paths in `systemd/` if yours differ.

## Repository layout

| Repository file | Installed to |
| --- | --- |
| `recorder/scanner-recorder.sh` | `/home/alex/scanner-recorder/scanner-recorder.sh` |
| `recorder/scanner-recorder.conf.example` | `/home/alex/.config/scanner-recorder.conf` (with your values) |
| `timeline/server.py` | `/home/alex/scanner-timeline/server.py` |
| `timeline/index.html` | `/home/alex/scanner-timeline/index.html` |
| `systemd/rdio-scanner.service` | `/etc/systemd/system/rdio-scanner.service` |
| `systemd/scanner-recorder.service` | `/etc/systemd/system/scanner-recorder.service` |
| `systemd/scanner-timeline.service` | `/etc/systemd/system/scanner-timeline.service` |

Created at runtime, not in the repository:

| Path | What it is |
| --- | --- |
| `/home/alex/scanner-archive/YYYY-MM-DD/` | Archived clips: `HHMMSS-n.wav` plus `HHMMSS-n.json` (time, duration, transcript) |
| `/home/alex/scanner-recordings/` | Recorder's temporary working folder |
| `/home/alex/scanner-recordings/failed/` | Clips whose Rdio Scanner upload failed |
| `/home/alex/whisper.cpp/` | whisper.cpp build and models |
| `/opt/rdio-scanner/` | Rdio Scanner program, database and settings |

The recorder writes each clip's `.wav` first and its `.json` last, so the timeline only ever shows complete recordings.

## Setup

1. **Tools and permissions:** `sudo apt install sox libsox-fmt-all curl ffmpeg unzip build-essential cmake git`, then `sudo usermod -aG audio alex` and log in again.
2. **Sound card:** check that `arecord -l` lists the USB card. Set the capture level with `alsamixer -c <card>` (F4 for capture) and test with `arecord -D plughw:CARD=Device,DEV=0 -f S16_LE -r 48000 -c 1 -d 5 test.wav`.
3. **Rdio Scanner:** unzip the latest linux-amd64 release into `/opt/rdio-scanner` and `chown` it to your user. Start it once, open `http://localhost:3000/admin` (first password `rdio-scanner`), change the password, create system 1 and talkgroup 1, create an API key and enable audio conversion.
4. **Recorder:** copy `recorder/scanner-recorder.sh` to `~/scanner-recorder/` and make it executable. Copy `recorder/scanner-recorder.conf.example` to `~/.config/scanner-recorder.conf` and fill in `DEVICE`, `KEY` and `FREQ`.
5. **whisper.cpp:** `git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp`, then in that folder run `cmake -B build`, `cmake --build build -j --config Release` and `sh ./models/download-ggml-model.sh base`.
6. **Timeline:** copy `timeline/server.py` and `timeline/index.html` to `~/scanner-timeline/`.
7. **Services:** copy the three files from `systemd/` to `/etc/systemd/system/`, run `sudo systemctl daemon-reload`, then `sudo systemctl enable --now rdio-scanner scanner-recorder scanner-timeline`.

### Recorder settings

`~/.config/scanner-recorder.conf` overrides the defaults at the top of the script. An empty `KEY` skips the Rdio upload and only archives.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `plughw:CARD=Device,DEV=0` | Sound card; list names with `arecord -L \| grep plughw` |
| `KEY` | empty | Rdio Scanner API key |
| `FREQ` | `118000000` | Monitored frequency in Hz |
| `THRESHOLD` | `2%` | Audio level that counts as a transmission |
| `HANG` | `2.0` | Seconds of silence that end a clip |
| `MIN_LEN` | `0.5` | Clips shorter than this are discarded |
| `ARCHIVE` | `~/scanner-archive` | Archive folder; must match the timeline's `ARCHIVE` |

### Timeline settings

Add these as `Environment=` lines in `scanner-timeline.service`, then run `sudo systemctl daemon-reload` and restart the service.

| Setting | Default | Purpose |
| --- | --- | --- |
| `WHISPER_LANG` | `en` | Transcription language; `auto` if German traffic is common |
| `WHISPER_MODEL` | `~/whisper.cpp/models/ggml-base.bin` | Full path to the model; `tiny` is faster, `small` more accurate |
| `WHISPER_THREADS` | `4` | CPU threads for transcription |
| `KEEP_DAYS` | `30` | Days of recordings to keep; `0` keeps everything |
| `PORT` | `8090` | Web port of the timeline |
| `ARCHIVE` | `~/scanner-archive` | Archive folder |

If the "waiting for transcription" count in the timeline header keeps growing, switch to a smaller model. An hour of radio traffic takes about 115 MB of disk space.

### Radio settings (UV-K5(8))

Set the channel to AM, turn off battery save (SAVE) and dual watch (TDR), and set squelch (SQL) just high enough that the radio is silent between transmissions.

## Use

| Address | Use it for |
| --- | --- |
| `http://adsb.local:8090` | Timeline: browse, search and replay transmissions with transcripts |
| `http://adsb.local:3000` | Rdio Scanner: live feed and call archive |
| `http://adsb.local:3000/admin` | Rdio Scanner settings |

In the timeline, pick a day at the top, click the activity strip to jump to a time, and click any transmission to play it. With "Play next automatically" ticked, playback continues through the day. Search filters the current day; tick "All days" to search the whole archive. Keys: Space plays or pauses, J next, K previous.

Transcripts often garble callsigns and numbers; use them to find clips, then listen.

## Managing services

| Command | Effect |
| --- | --- |
| `systemctl status <name>` | Show whether it runs and its last log lines |
| `journalctl -u <name> -f` | Follow its log live |
| `sudo systemctl restart <name>` | Restart, e.g. after changing settings |
| `sudo systemctl stop <name>` | Stop until the next reboot |
| `sudo systemctl disable <name>` | Don't start at boot (`enable` to undo) |
| `sudo systemctl reset-failed <name>` | Clear a "start request repeated too quickly" failure |

To update, copy the changed file from the repository over the installed one and restart its service.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `arecord -l`: "no soundcards found" | User not in the `audio` group yet. Run `sudo usermod -aG audio alex`, then log in again |
| Recorder: "audio open error" or "No audio from …" | Wrong `DEVICE`. Use the card's name (`plughw:CARD=Device,DEV=0`); card numbers can change |
| "Device or resource busy" | The desktop audio system holds the card. In `pavucontrol`, Configuration tab, set the USB card's profile to Off |
| `rec` hangs and ignores Ctrl+C | SoX can't read this card directly; the recorder uses `arecord` for that reason. End a stuck process with `pkill -9 -x rec` |
| Service status `203/EXEC` | `ExecStart` path in the unit file doesn't match where the script is |
| Rdio upload: "no audio" | Empty clips, usually from a wrong `DEVICE` |
| Many short junk clips | Squelch too open or threshold too low. Tighten the radio's SQL or raise `THRESHOLD` |
| Pilot replies split from the controller | Raise `HANG` to `3.0` |
| First syllable cut off | Battery save is on in the radio |
| Timeline: "Transcription off: whisper.cpp not found" | `whisper-cli` or the model is missing at the configured paths |

The recorder estimates each clip's start time from when it ended, so times can be off by a second or two.

## Future extension: ADS-B receiver

The same laptop is meant to also run an ADS-B receiver once a working SDR stick is in place (the original NooElec NESDR SMArt is defective).

- **Hardware:** a 1090 MHz stick with built-in filter and amplifier (FlightAware Pro Stick Plus or AirNav FlightStick), or a general-purpose RTL-SDR (NESDR SMArt v5, RTL-SDR Blog V4). The NooElec kit antenna fits via SMA; collapse it to about 6.9 cm and mount it vertically, as high as possible.
- **Software:** install [readsb](https://github.com/wiedehopf/readsb) with `sudo bash -c "$(wget -O - https://github.com/wiedehopf/adsb-scripts/raw/master/readsb-install.sh)"`, set the location with `sudo readsb-set-location <lat> <lon>`, then install [tar1090](https://github.com/wiedehopf/tar1090) with `sudo bash -c "$(wget -nv -O - https://github.com/wiedehopf/tar1090/raw/master/install.sh)"`. Reboot so the DVB-TV driver blacklist takes effect.
- **Use:** the map will be at `http://adsb.local/tar1090`; port 80 doesn't clash with Rdio Scanner (3000) or the timeline (8090).
- **Check:** `lsusb` should list the stick as Realtek RTL2838.

## Legal note

In Germany, listening to aviation radio is generally tolerated, but passing on what you hear is prohibited (§ 5 TDDDG). Keep the web interfaces on your home network and keep this repository free of recordings.
