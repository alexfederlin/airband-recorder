# Airband recorder

Records an airport frequency from a scanner, splits it into one clip per transmission, and makes the recordings searchable in the browser.

- **Recorder** (`recorder/`): captures the scanner audio from a USB sound card, cuts it into clips, archives each clip and uploads it to [Rdio Scanner](https://github.com/chuot/rdio-scanner).
- **Timeline** (`timeline/`): a small Python server (standard library only) that transcribes archived clips with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and serves a day-by-day timeline with search and continuous playback.
- **Rdio Scanner** (optional): live feed with near-instant playback, mobile apps and its own call archive, run from its official Docker image. Off by default; see [Rdio Scanner](#rdio-scanner-optional).

All of them run as containers, started with one `docker compose` command.

Built for a ThinkPad Yoga 12 running Xubuntu (hostname `adsb`), a Quansheng UV-K5(8) and a C-Media USB sound card.

## Repository layout

| Path | What it is |
| --- | --- |
| `docker-compose.yml` | Starts all three containers |
| `.env.example` | Settings template; copy to `.env` |
| `recorder/` | Recorder script and its `Dockerfile` |
| `timeline/` | Timeline server, web page and its `Dockerfile` (builds whisper.cpp and downloads the speech model) |
| `testset/` | Separate tool to compare Whisper models on hand-transcribed clips (see `testset/README.md`) |
| `migrate-from-bare-metal.sh` | One-time move from the systemd install to Docker |
| `systemd/` | Unit files for running without Docker (see the end of this file) |

Created at runtime, not in the repository:

| Path | What it is |
| --- | --- |
| `data/archive/YYYY-MM-DD/` | Archived clips: `HHMMSS-n.wav` plus `HHMMSS-n.json` (time, duration, transcript, model). `ARCHIVE_DIR` can put it elsewhere |
| `data/recordings/` | Recorder's temporary working folder |
| `data/recordings/failed/` | Clips whose Rdio Scanner upload failed |
| `data/rdio-scanner/` | Rdio Scanner database and settings (`RDIO_DATA` can put it elsewhere) |
| `models/` | Extra speech models for the timeline, such as an ATC fine-tune from the test set |

The recorder writes each clip's `.wav` first and its `.json` last, so the timeline only ever shows complete recordings.

## Setup

1. **Docker:** `sudo apt install docker.io docker-compose-v2 alsa-utils git`, then `sudo usermod -aG docker $USER` and log in again.
2. **Sound card:** check that `arecord -l` lists the USB card. Set the capture level with `alsamixer -c <card>` (F4 for capture) and test with `arecord -D plughw:CARD=Device,DEV=0 -f S16_LE -r 48000 -c 1 -d 5 test.wav`. The desktop audio system must not hold the card: in `pavucontrol`, Configuration tab, set the USB card's profile to Off.
3. **Get the code and settings:** `git clone` this repository, `cd` into it, run `cp .env.example .env` and fill in `DEVICE` and `FREQ`. Leave `KEY` empty for now. Run `mkdir -p data/rdio-scanner models` so these folders belong to you rather than root.
4. **Start:** `docker compose up -d --build`. The first build compiles whisper.cpp and downloads the speech model, which takes a while on the laptop.

The containers restart automatically, including after a reboot.

### Rdio Scanner (optional)

The timeline covers the archive, search and live listening (it picks up new clips every 3 seconds). Rdio Scanner
adds a near-instant live feed for many listeners and its iOS/Android apps, at the cost of a second copy of every
clip and CPU for converting them. To turn it on:

1. Add `COMPOSE_PROFILES=rdio` to `.env` and run `docker compose up -d`.
2. Open `http://adsb.local:3000/admin` (first password `rdio-scanner`), change the password, create system 1 and
   talkgroup 1, create an API key and enable audio conversion.
3. Put the key into `KEY` in `.env` and run `docker compose up -d` again.

To turn it off again, remove `COMPOSE_PROFILES=rdio` and run `docker compose up -d --remove-orphans`. Its database
stays in `data/rdio-scanner/`. If `KEY` is still set, the recorder notes once that Rdio Scanner isn't reachable and
just archives.

### Moving from the systemd install

If the recorder ran on bare metal before (`systemd/`), the Docker setup can take over its data:

1. Stop the old services: `sudo systemctl disable --now rdio-scanner scanner-recorder scanner-timeline`.
2. In the repository folder, as your normal user: `./migrate-from-bare-metal.sh`. It writes `.env` from
   `~/.config/scanner-recorder.conf`, keeps the recordings where they are (`ARCHIVE_DIR=~/scanner-archive`),
   copies Rdio Scanner's database (systems, talkgroups, API key, calls) from `/opt/rdio-scanner/` to
   `data/rdio-scanner/`, and copies the ATC model from `testset/models/` to `models/` if the test set made it.
   It only copies, so the old files stay as a backup.
3. `docker compose up -d --build` (skip step 3 above). If you turn Rdio Scanner on, it keeps its settings and API
   key, so only step 1 of [Rdio Scanner](#rdio-scanner-optional) is needed.

Once everything works, the old `/opt/rdio-scanner/`, `~/scanner-timeline/`, `~/scanner-recorder/` and
`~/whisper.cpp/` can go.

### Recorder settings

Set these in `.env`, then run `docker compose up -d`. An empty `KEY` skips the Rdio upload and only archives.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `plughw:CARD=Device,DEV=0` | Sound card; list names with `arecord -L \| grep plughw` |
| `KEY` | empty | Rdio Scanner API key (only with Rdio Scanner turned on) |
| `FREQ` | `118000000` | Monitored frequency in Hz |
| `THRESHOLD` | `2%` | Audio level that counts as a transmission |
| `HANG` | `2.0` | Seconds of silence that end a clip |
| `MIN_LEN` | `0.5` | Clips shorter than this are discarded |
| `SYSTEM`, `TALKGROUP` | `1` | Rdio Scanner system and talkgroup IDs |
| `TZ` | `Europe/Berlin` | Time zone for clip times and day folders |
| `ARCHIVE_DIR` | `./data/archive` | Where clips and transcripts are kept (full path) |
| `RDIO_DATA` | `./data/rdio-scanner` | Rdio Scanner's database folder |

### Timeline settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `WHISPER_LANG` | `en` | Transcription language; `auto` if German traffic is common |
| `WHISPER_MODEL_NAME` | `base` | Speech model built into the image; `tiny` is faster, `small` more accurate. Needs `docker compose up -d --build` after a change |
| `WHISPER_MODEL` | built-in model | A model file in `./models` instead, as `/extra-models/<file>`, e.g. `/extra-models/ggml-jlvdoorn--whisper-small-atco2-asr.bin` (the ATC fine-tune that won the test set). Needs `docker compose up -d` after a change |
| `WHISPER_PROMPT` | built-in ATC prompt | Text that steers Whisper towards local names, callsigns and phrasing; `none` turns it off. Try variants in the test set's Prompt lab first. Put it in double quotes |
| `RETRANSCRIBE` | `1` | Redo transcripts made with another model, language or prompt, in the background after new clips; `0` keeps them |
| `WHISPER_THREADS` | `4` | CPU threads for transcription |
| `KEEP_DAYS` | `30` | Days of recordings to keep; `0` keeps everything |

If the "waiting for transcription" count in the timeline header keeps growing, switch to a smaller model. An hour of radio traffic takes about 115 MB of disk space.

Each transcript records the model and language that made it. After a change, the header shows "N older clips to
re-transcribe" and the timeline works through the archive newest first, one old clip at a time between new ones,
so live clips never wait behind the backlog. On the laptop's dual-core CPU a Small model takes a while for a full
archive; the old transcript stays visible until its replacement is ready.

### Radio settings (UV-K5(8))

Set the channel to AM, turn off battery save (SAVE) and dual watch (TDR), and set squelch (SQL) just high enough that the radio is silent between transmissions.

## Use

| Address | Use it for |
| --- | --- |
| `http://adsb.local:8090` | Timeline: browse, search and replay transmissions with transcripts |
| `http://adsb.local:3000` | Rdio Scanner, if turned on: live feed and call archive |
| `http://adsb.local:3000/admin` | Rdio Scanner settings |

In the timeline, pick a day at the top, click the activity strip to jump to a time, and click any transmission to play it. With "Play next automatically" ticked, playback continues through the day; at the end of today it waits and plays each new transmission as it arrives, a few seconds after it ends (the recorder's `HANG` of silence plus up to 3 s). Search filters the current day; tick "All days" to search the whole archive. Keys: Space plays or pauses, J next, K previous.

Transcripts often garble callsigns and numbers; use them to find clips, then listen.

### Callsigns

German registrations are recognised in the transcripts and shown in letters: "Delta Echo Alpha Romeo November"
becomes **D-EARN**, both in the transcript and in the callsign column next to it. The short form ("Delta Romeo
November") and a callsign with one letter lost or misheard become the full callsign when that is spoken in the same
transmission; otherwise a short form shows as D-RN. Searching for D-EARN or D-RN finds the spoken forms too, also
across all days.

The bar under the activity strip lists the day's callsigns with the number of transmissions each appears in, most
frequent first; a short form counts towards the day's full callsign ending in the same letters. Click a callsign to
highlight its transmissions in the list and on the strip (the others are dimmed); click it again or press Esc to clear.

Whisper sometimes hears a letter word as something else, e.g. "homie" for Romeo. Common cases are built in; add
your own to `config/letters.txt` (copy `config/letters.txt.example`), one `heard = letter word` per line, then reload
the page. These words only count as letters inside a spelled-out callsign; a word you map to Delta
(`welter = Delta`) can also start one. For names that are always wrong (for
example the tower's), a better prompt helps more: see `WHISPER_PROMPT` and the test set's Prompt lab.

### Aircraft nearby (ADS-B)

With the airport's position in `.env` (`AIRPORT_LAT`, `AIRPORT_LON`; Mönchengladbach is 51.2303, 6.5047), the
timeline asks free ADS-B networks which aircraft are near the airport as soon as the recorder archives a
transmission, and stores them in the clip's `.json`:

- **Shown with each transmission:** callsign or registration, type, altitude and distance from the airport, for
  aircraft within `ADSB_RADIUS_KM` and below `ADSB_MAX_ALT_FT`. Aircraft on the ground count only within
  `ADSB_GROUND_KM`, so traffic at Düsseldorf stays out. The aircraft the transmission probably addressed comes first,
  marked with an arrow, and its callsign joins the callsign column and the day's callsign bar.
- **Better recognition:** their callsigns go into Whisper's prompt for that clip, the way they are spoken ("Delta
  Echo Kilo Lima Mike" for D-EKLM, "Lufthansa four alpha bravo" for DLH4AB). Afterwards, short forms ("Delta Lima Mike")
  and callsigns with a misheard letter resolve to the nearby aircraft, and airline callsigns are found in the
  transcript ("Lufthansa four alpha bravo" or "Lufthansa 4AB", shown in bold; searching DLH4AB finds them).

All networks are asked at once and their answers merged per aircraft, since one may know the callsign and another
the registration:

| Network | Notes |
| --- | --- |
| [adsb.lol](https://adsb.lol) | Free, no key |
| [adsb.fi](https://adsb.fi) | Free for personal, non-commercial use; asks to be credited (the page does). One request per second |
| [OpenSky](https://opensky-network.org) | Callsigns but no registrations. Anonymous use allows 400 requests a day, so the timeline spreads them over the day and reuses the last answer in between |
| local readsb | `ADSB_LOCAL_URL`, once there is an SDR stick on the laptop; asked first |

airplanes.live requires registration by email, so it isn't included. The networks only know the present, so only
transmissions recorded while the feature is on get a snapshot; older ones show none. Airlines are matched by their
radio name: common ones are built in, add more in `config/airlines.txt` (copy `config/airlines.txt.example`).

| Setting | Default | Purpose |
| --- | --- | --- |
| `AIRPORT_LAT`, `AIRPORT_LON` | empty (off) | The airport's position in decimal degrees |
| `ADSB_RADIUS_KM` | `20` | Aircraft within this distance count |
| `ADSB_MAX_ALT_FT` | `5000` | Aircraft above this altitude don't |
| `ADSB_GROUND_KM` | `3` | Aircraft on the ground count only this close |
| `ADSB_SOURCES` | `adsb.lol,adsb.fi,opensky` | Networks to ask |
| `ADSB_LOCAL_URL` | empty | A local readsb's `aircraft.json` |

## Managing the containers

Run these in the repository folder. The service names are `recorder`, `timeline` and (if turned on) `rdio-scanner`.

| Command | Effect |
| --- | --- |
| `docker compose ps` | Show which containers run |
| `docker compose logs -f <name>` | Follow a container's log live |
| `docker compose restart <name>` | Restart one container |
| `docker compose up -d` | Apply changed settings from `.env` |
| `docker compose down` | Stop everything (recordings in `./data` are kept) |
| `git pull && docker compose up -d --build` | Update to the latest code |
| `docker compose pull rdio-scanner && docker compose up -d` | Update Rdio Scanner |

Files the recorder and timeline write are owned by root, including new ones in `ARCHIVE_DIR`; use `sudo` to delete them by hand.

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
| Timeline: "Transcription off" with `WHISPER_MODEL` set | The file isn't in `./models`; check the name with `ls models/` |
| Rdio Scanner: "permission denied" on its database | `RDIO_DATA` must belong to the user in `UID`/`GID` (default 1000); fix with `sudo chown -R $(id -u):$(id -g) data/rdio-scanner` |
| Build: "Could not resolve host" | Builds already use the host's network; check the laptop's own internet connection |

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
