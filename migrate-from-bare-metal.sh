#!/usr/bin/env bash
# One-time move from the systemd install to Docker. Run it in the repository
# folder as your normal user (not root), after stopping the systemd services:
#
#   ./migrate-from-bare-metal.sh
#
# It only copies and never deletes: the bare-metal files stay as a backup.
#   - ~/.config/scanner-recorder.conf  -> .env (device, API key, frequency, ...)
#   - ~/scanner-archive                -> used in place (ARCHIVE_DIR in .env)
#   - /opt/rdio-scanner/rdio-scanner.db -> data/rdio-scanner/ (systems, talkgroups,
#                                          API keys and calls)
#   - testset/models/<ATC model>       -> models/ (WHISPER_MODEL in .env)
set -euo pipefail
cd "$(dirname "$0")"

CONF="${CONF:-$HOME/.config/scanner-recorder.conf}"
ARCHIVE_SRC="${ARCHIVE_SRC:-$HOME/scanner-archive}"
RDIO_SRC="${RDIO_SRC:-/opt/rdio-scanner}"
ATC_MODEL="${ATC_MODEL:-ggml-jlvdoorn--whisper-small-atco2-asr.bin}"

if [ "$(id -u)" = 0 ]; then
    echo "Run this as your normal user, not with sudo."
    exit 1
fi

# --- 1. The systemd services must be stopped ------------------------------------
for s in rdio-scanner scanner-recorder scanner-timeline; do
    if systemctl is-active --quiet "$s" 2>/dev/null; then
        echo "The systemd service $s is still running. Stop and disable them first:"
        echo "  sudo systemctl disable --now rdio-scanner scanner-recorder scanner-timeline"
        exit 1
    fi
done

mkdir -p data/rdio-scanner data/recordings models

# --- 2. Rdio Scanner database -------------------------------------------------------
if [ -f data/rdio-scanner/rdio-scanner.db ]; then
    echo "Rdio Scanner: data/rdio-scanner/rdio-scanner.db exists already, not copied."
elif [ -f "$RDIO_SRC/rdio-scanner.db" ]; then
    # -wal/-shm too, in case the service didn't shut down cleanly
    cp -p "$RDIO_SRC"/rdio-scanner.db* data/rdio-scanner/
    echo "Rdio Scanner: copied the database from $RDIO_SRC (the original stays there)."
else
    echo "Rdio Scanner: no database at $RDIO_SRC/rdio-scanner.db; Docker starts with an empty one."
fi

# --- 3. ATC model from the test set ---------------------------------------------------
if [ -f "models/$ATC_MODEL" ]; then
    echo "Model: models/$ATC_MODEL exists already."
elif [ -f "testset/models/$ATC_MODEL" ]; then
    cp "testset/models/$ATC_MODEL" models/
    echo "Model: copied $ATC_MODEL from testset/models/."
else
    echo "Model: testset/models/$ATC_MODEL not found; the timeline uses its built-in model."
fi

# --- 4. .env -------------------------------------------------------------------------------
if [ -e .env ]; then
    echo ".env exists already, left as it is. Compare it with .env.example."
else
    if [ ! -f "$CONF" ]; then
        echo "No $CONF found. Copy .env.example to .env and fill it in by hand."
        exit 1
    fi
    # Read the recorder settings in a subshell, so nothing else from the file leaks in
    settings=$(
        set +u
        # shellcheck source=/dev/null
        source "$CONF"
        for v in DEVICE KEY FREQ THRESHOLD HANG MIN_LEN SYSTEM TALKGROUP; do
            [ -n "${!v-}" ] && printf '%s=%s\n' "$v" "${!v}"
        done
        true
    )
    tz=$(cat /etc/timezone 2>/dev/null || echo Europe/Berlin)
    {
        echo "# Created by migrate-from-bare-metal.sh from $CONF."
        echo "# More settings and explanations: .env.example. Never commit this file."
        echo "TZ=$tz"
        echo "$settings"
        [ -d "$ARCHIVE_SRC" ] && echo "ARCHIVE_DIR=$ARCHIVE_SRC"
        echo "WHISPER_LANG=en"
        [ -f "models/$ATC_MODEL" ] && echo "WHISPER_MODEL=/extra-models/$ATC_MODEL"
        echo "RETRANSCRIBE=1"
    } > .env
    chmod 600 .env
    echo ".env: written from $CONF (the API key is in it, so it's readable only by you)."
fi

cat <<'EOF'

Next:
  docker compose up -d --build        # first build compiles whisper.cpp, takes a while
  docker compose logs -f recorder     # should say "Listening on ..." and log clips
  docker compose logs -f timeline     # transcription and re-transcription progress

Then check http://adsb.local:8090 (timeline) and http://adsb.local:3000 (Rdio Scanner).
Clips whose Rdio upload failed on bare metal are still in ~/scanner-recordings/failed/.
EOF
