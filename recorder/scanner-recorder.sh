#!/usr/bin/env bash
# scanner-recorder.sh
# Records scanner audio one transmission at a time, archives each clip for the
# timeline page and (optionally) uploads it to Rdio Scanner.
set -u

# --- Default settings ---------------------------------------------------------
# Don't edit these here. Put your own values in ~/.config/scanner-recorder.conf
# so they survive future updates of this script.
DEVICE="plughw:CARD=Device,DEV=0"    # USB sound card (list them with: arecord -L | grep plughw)
API="http://localhost:3000/api/call-upload"
KEY=""                               # Rdio Scanner API key; leave empty to skip uploading
SYSTEM=1                             # System ID in Rdio Scanner
TALKGROUP=1                          # Talkgroup ID in Rdio Scanner
FREQ=118000000                       # Monitored frequency in Hz
THRESHOLD="2%"                       # Audio level that counts as a transmission
HANG="2.0"                           # Seconds of silence that end a recording
MIN_LEN="0.5"                        # Discard clips shorter than this (clicks, blips)
DIR="$HOME/scanner-recordings"       # Temporary working folder
ARCHIVE="$HOME/scanner-archive"      # Clips for the timeline page, one folder per day
# ------------------------------------------------------------------------------

CONFIG="${CONFIG:-$HOME/.config/scanner-recorder.conf}"
if [[ -f "$CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG"
fi

mkdir -p "$DIR/failed" "$ARCHIVE"

upload_enabled() {
  [[ -n "$KEY" && "$KEY" != "PASTE-YOUR-API-KEY-HERE" ]]
}

upload() {
  local file="$1" start="$2" resp
  resp=$(curl -s --max-time 30 "$API" \
    -F "audio=@${file};type=audio/wav" \
    -F "audioName=$(basename "$file")" \
    -F "audioType=audio/wav" \
    -F "dateTime=$(date -u -d "@${start}" +%Y-%m-%dT%H:%M:%SZ)" \
    -F "frequency=${FREQ}" \
    -F "key=${KEY}" \
    -F "system=${SYSTEM}" \
    -F "talkgroup=${TALKGROUP}")
  if [[ "$resp" == *"success"* ]]; then
    rm -f "$file"
  else
    echo "Upload failed for $(basename "$file"): $resp" >&2
    mv "$file" "$DIR/failed/" 2>/dev/null
  fi
}

echo "Listening on $DEVICE ..."
if upload_enabled; then
  echo "Uploading to Rdio Scanner at $API"
else
  echo "No Rdio Scanner API key set, clips are only archived"
fi

count=0
while true; do
  tmp="$DIR/current.wav"
  rm -f "$tmp"
  loop_start=$(date +%s)

  # arecord captures the audio; sox waits until the level rises above THRESHOLD,
  # then records until HANG seconds of silence. When sox exits, arecord stops too.
  arecord -q -D "$DEVICE" -f S16_LE -r 16000 -c 1 -t raw \
    | sox -q -t raw -r 16000 -e signed -b 16 -c 1 - "$tmp" \
        silence 1 0.1 "$THRESHOLD" 1 "$HANG" "$THRESHOLD"
  status=("${PIPESTATUS[@]}")

  # 16 kHz, 16-bit mono = 32000 bytes per second, plus a 44-byte WAV header
  size=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
  min_bytes=$(awk "BEGIN { printf \"%d\", 44 + 32000 * $MIN_LEN }")

  if [[ ${status[1]} -ne 0 ]] || (( size < min_bytes )); then
    rm -f "$tmp"
    # An empty result within a moment of starting means no audio is arriving at all
    if (( $(date +%s) - loop_start < 2 )); then
      echo "No audio from $DEVICE (wrong device name or card busy?), retrying in 5 s" >&2
      sleep 5
    fi
    continue
  fi

  end=$(date +%s)
  dur=$(soxi -D "$tmp" 2>/dev/null || echo 0)
  start=$(( end - ${dur%.*} ))
  count=$(( count + 1 ))

  file="$DIR/call-$(date -d "@${start}" +%Y%m%d-%H%M%S)-${count}.wav"
  mv "$tmp" "$file"

  # Archive a copy for the timeline page. The .json is written last, so the
  # timeline only ever sees complete clips.
  day=$(date -d "@${start}" +%Y-%m-%d)
  base="$(date -d "@${start}" +%H%M%S)-${count}"
  mkdir -p "$ARCHIVE/$day"
  cp "$file" "$ARCHIVE/$day/$base.wav"
  printf '{"start": %d, "duration": %s, "frequency": %d}\n' "$start" "$dur" "$FREQ" \
    > "$ARCHIVE/$day/$base.json.tmp" \
    && mv "$ARCHIVE/$day/$base.json.tmp" "$ARCHIVE/$day/$base.json"

  if upload_enabled; then
    # Upload in the background so recording resumes immediately
    upload "$file" "$start" &
  else
    rm -f "$file"
  fi
done
