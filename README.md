# Airband recorder

Records an airport frequency from a scanner, splits it into one clip per transmission, and makes the recordings searchable in the browser.

- **Recorder** (`recorder/`): captures the scanner audio from a USB sound card, cuts it into clips, archives each clip and uploads it to [Rdio Scanner](https://github.com/chuot/rdio-scanner).
- **Timeline** (`timeline/`): a small Python server (standard library only) that transcribes archived clips with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and serves a day-by-day timeline with search and continuous playback.
- **Rdio Scanner**: live feed and call archive, run from its official Docker image.

All three run as containers, started with one `docker compose` command.

Built for a ThinkPad Yoga 12 running Xubuntu (hostname `adsb`), a Quansheng UV-K5(8) and a C-Media USB sound card.

## Repository layout

| Path | What it is |
| --- | --- |
| `docker-compose.yml` | Starts all three containers |
| `.env.example` | Settings template; copy to `.env` |
| `recorder/` | Recorder script and its `Dockerfile` |
| `timeline/` | Timeline server, web page and its `Dockerfile` (builds whisper.cpp and downloads the speech model) |
| `systemd/` | Unit files for running without Docker (see the end of this file) |

Created at runtime in `./data`, not in the repository:

| Path | What it is |
| --- | --- |
| `data/archive/YYYY-MM-DD/` | Archived clips: `HHMMSS-n.wav` plus `HHMMSS-n.json` (time, duration, transcript) |
| `data/recordings/` | Recorder's temporary working folder |
| `data/recordings/failed/` | Clips whose Rdio Scanner upload failed |
| `data/rdio-scanner/` | Rdio Scanner database and settings |

The recorder writes each clip's `.wav` first and its `.json` last, so the timeline only ever shows complete recordings.

## Setup

1. **Docker:** `sudo apt install docker.io docker-compose-v2 alsa-utils git`, then `sudo usermod -aG docker $USER` and log in again.
2. **Sound card:** check that `arecord -l` lists the USB card. Set the capture level with `alsamixer -c <card>` (F4 for capture) and test with `arecord -D plughw:CARD=Device,DEV=0 -f S16_LE -r 48000 -c 1 -d 5 test.wav`. The desktop audio system must not hold the card: in `pavucontrol`, Configuration tab, set the USB card's profile to Off.
3. **Get the code and settings:** `git clone` this repository, `cd` into it, run `cp .env.example .env` and fill in `DEVICE` and `FREQ`. Leave `KEY` empty for now.
4. **Start:** `docker compose up -d --build`. The first build compiles whisper.cpp and downloads the speech model, which takes a while on the laptop.
5. **Rdio Scanner:** open `http://adsb.local:3000/admin` (first password `rdio-scanner`), change the password, create system 1 and talkgroup 1, create an API key and enable audio conversion. Put the key into `KEY` in `.env` and run `docker compose up -d` again.

The containers restart automatically, including after a reboot.

### Recorder settings

Set these in `.env`, then run `docker compose up -d`. An empty `KEY` skips the Rdio upload and only archives.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `plughw:CARD=Device,DEV=0` | Sound card; list names with `arecord -L \| grep plughw` |
| `KEY` | empty | Rdio Scanner API key |
| `FREQ` | `118000000` | Monitored frequency in Hz |
| `THRESHOLD` | `2%` | Audio level that counts as a transmission |
| `HANG` | `2.0` | Seconds of silence that end a clip |
| `MIN_LEN` | `0.5` | Clips shorter than this are discarded |
| `TZ` | `Europe/Berlin` | Time zone for clip times and day folders |

### Timeline settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `WHISPER_LANG` | `en` | Transcription language; `auto` if German traffic is common |
| `WHISPER_MODEL_NAME` | `base` | Speech model built into the image; `tiny` is faster, `small` more accurate. Needs `docker compose up -d --build` after a change |
| `WHISPER_THREADS` | `4` | CPU threads for transcription |
| `KEEP_DAYS` | `30` | Days of recordings to keep; `0` keeps everything |

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

## Managing the containers

Run these in the repository folder. The service names are `recorder`, `timeline` and `rdio-scanner`.

| Command | Effect |
| --- | --- |
| `docker compose ps` | Show which containers run |
| `docker compose logs -f <name>` | Follow a container's log live |
| `docker compose restart <name>` | Restart one container |
| `docker compose up -d` | Apply changed settings from `.env` |
| `docker compose down` | Stop everything (recordings in `./data` are kept) |
| `git pull && docker compose up -d --build` | Update to the latest code |
| `docker compose pull rdio-scanner && docker compose up -d` | Update Rdio Scanner |

Files in `./data` are owned by root because the containers write them; use `sudo` to delete them by hand.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Recorder: "audio open error" or "No audio from …" | Wrong `DEVICE`. Use the card's name (`plughw:CARD=Device,DEV=0`); card numbers can change |
| "Device or resource busy" | The desktop audio system holds the card. In `pavucontrol`, Configuration tab, set the USB card's profile to Off |
| Recorder starts before the USB card is plugged in | Plug it in, then `docker compose restart recorder` |
| Rdio upload: "no audio" | Empty clips, usually from a wrong `DEVICE` |
| Many short junk clips | Squelch too open or threshold too low. Tighten the radio's SQL or raise `THRESHOLD` |
| Pilot replies split from the controller | Raise `HANG` to `3.0` |
| First syllable cut off | Battery save is on in the radio |
| Timeline: "Transcription off: whisper.cpp not found" | The image build failed partway; run `docker compose build --no-cache timeline` and check its output |
| Timeline crashes with "Illegal instruction" | The image was built on a different computer. Build it on the laptop itself |

The recorder estimates each clip's start time from when it ended, so times can be off by a second or two.

## Running without Docker

The scripts still run directly on the host. Install `sox libsox-fmt-all curl ffmpeg build-essential cmake git`, add your user to the `audio` group, build whisper.cpp in `~/whisper.cpp`, unzip Rdio Scanner into `/opt/rdio-scanner`, copy `recorder/scanner-recorder.sh` to `~/scanner-recorder/` and `timeline/` to `~/scanner-timeline/`, and put your settings in `~/.config/scanner-recorder.conf` (template: `recorder/scanner-recorder.conf.example`). The unit files in `systemd/` start everything at boot; adjust the user name and paths in them, copy them to `/etc/systemd/system/` and run `sudo systemctl enable --now rdio-scanner scanner-recorder scanner-timeline`.

## Future extension: ADS-B receiver

The same laptop is meant to also run an ADS-B receiver once a working SDR stick is in place (the original NooElec NESDR SMArt is defective).

- **Hardware:** a 1090 MHz stick with built-in filter and amplifier (FlightAware Pro Stick Plus or AirNav FlightStick), or a general-purpose RTL-SDR (NESDR SMArt v5, RTL-SDR Blog V4). The NooElec kit antenna fits via SMA; collapse it to about 6.9 cm and mount it vertically, as high as possible.
- **Software:** install [readsb](https://github.com/wiedehopf/readsb) with `sudo bash -c "$(wget -O - https://github.com/wiedehopf/adsb-scripts/raw/master/readsb-install.sh)"`, set the location with `sudo readsb-set-location <lat> <lon>`, then install [tar1090](https://github.com/wiedehopf/tar1090) with `sudo bash -c "$(wget -nv -O - https://github.com/wiedehopf/tar1090/raw/master/install.sh)"`. Reboot so the DVB-TV driver blacklist takes effect.
- **Use:** the map will be at `http://adsb.local/tar1090`; port 80 doesn't clash with Rdio Scanner (3000) or the timeline (8090).
- **Check:** `lsusb` should list the stick as Realtek RTL2838.

## Legal note

In Germany, listening to aviation radio is generally tolerated, but passing on what you hear is prohibited (§ 5 TDDDG). Keep the web interfaces on your home network and keep this repository free of recordings.
