#!/usr/bin/env python3
"""
Scanner timeline server.

Serves the timeline page and the archived recordings, and transcribes new
recordings in the background with whisper.cpp. Uses only the Python standard
library. Settings can be changed with environment variables (see below).
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HOME = Path.home()

# --- Settings (override with environment variables) --------------------------
ARCHIVE = Path(os.environ.get("ARCHIVE", HOME / "scanner-archive"))
PORT = int(os.environ.get("PORT", "8090"))
WHISPER_BIN = os.environ.get("WHISPER_BIN", str(HOME / "whisper.cpp/build/bin/whisper-cli"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", str(HOME / "whisper.cpp/models/ggml-base.bin"))
WHISPER_LANG = os.environ.get("WHISPER_LANG", "en")      # "en", "de" or "auto"
WHISPER_THREADS = os.environ.get("WHISPER_THREADS", "4")
WHISPER_PROMPT = os.environ.get(
    "WHISPER_PROMPT",
    "Air traffic control radio. Runway, taxi, holding point, cleared, QNH, "
    "squawk, wind, contact tower, report, final, downwind.",
)
KEEP_DAYS = int(os.environ.get("KEEP_DAYS", "30"))         # 0 = keep forever
# 1 = transcripts made with another model or language are redone in the
# background, after new clips (e.g. the whole archive after a model change)
RETRANSCRIBE = os.environ.get("RETRANSCRIBE", "1") == "1"
# ------------------------------------------------------------------------------

# Stored with each transcript, so a model change can be detected later
MODEL_TAG = f"{Path(WHISPER_MODEL).name}|{WHISPER_LANG}"

PAGE = Path(__file__).resolve().parent / "index.html"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.wav$")

# Text whisper tends to invent for noise or silence
NOISE_TAGS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
HALLUCINATIONS = {
    "", "you", "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "thanks for watching.", "bye.", "bye", ".",
}

STATE = {"transcription": "starting", "pending": 0, "redo": 0, "error": None}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# --- Archive access -----------------------------------------------------------

def load_meta(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_meta(path, meta):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp, path)


def day_dirs(newest_first=True):
    if not ARCHIVE.is_dir():
        return []
    dirs = [d for d in ARCHIVE.iterdir() if d.is_dir() and DATE_RE.match(d.name)]
    return sorted(dirs, key=lambda d: d.name, reverse=newest_first)


def clip_status(meta):
    if meta.get("transcript_error"):
        return "error"
    if "transcript" in meta:
        return "done"
    return "pending"


def clips_for_day(day):
    folder = ARCHIVE / day
    clips = []
    if not folder.is_dir():
        return clips
    for meta_path in folder.glob("*.json"):
        meta = load_meta(meta_path)
        wav = meta_path.with_suffix(".wav")
        if not meta or not wav.exists():
            continue
        clips.append({
            "id": f"{day}/{wav.name}",
            "start": meta.get("start", 0),
            "duration": meta.get("duration", 0),
            "frequency": meta.get("frequency"),
            "transcript": meta.get("transcript"),
            "status": clip_status(meta),
        })
    clips.sort(key=lambda c: c["start"])
    return clips


def list_days():
    days = []
    for folder in day_dirs():
        clips = clips_for_day(folder.name)
        days.append({
            "date": folder.name,
            "count": len(clips),
            "seconds": round(sum(c["duration"] or 0 for c in clips), 1),
        })
    return days


def search(query, limit=300):
    q = query.lower()
    results = []
    for folder in day_dirs():
        for clip in reversed(clips_for_day(folder.name)):
            if clip["transcript"] and q in clip["transcript"].lower():
                results.append(dict(clip, date=folder.name))
                if len(results) >= limit:
                    return results
    return results


# --- Background transcription ------------------------------------------------

def find_pending():
    """Returns (new, redo): clips without a transcript, and (with RETRANSCRIBE)
    clips whose transcript came from another model or language. Newest first."""
    new, redo = [], []
    for folder in day_dirs():
        for meta_path in folder.glob("*.json"):
            meta = load_meta(meta_path)
            if meta is None or not meta_path.with_suffix(".wav").exists():
                continue
            status = clip_status(meta)
            if status == "pending":
                new.append((meta.get("start", 0), meta_path))
            elif RETRANSCRIBE and status == "done" and meta.get("transcript_model") != MODEL_TAG:
                redo.append((meta.get("start", 0), meta_path))
    new.sort(key=lambda p: p[0], reverse=True)
    redo.sort(key=lambda p: p[0], reverse=True)
    return [p[1] for p in new], [p[1] for p in redo]


def clean_text(text):
    text = NOISE_TAGS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in HALLUCINATIONS:
        return ""
    if WHISPER_PROMPT and text.strip(" .").lower() in WHISPER_PROMPT.lower():
        return ""   # whisper sometimes just repeats the prompt on noise
    return text


def transcribe(wav):
    cmd = [
        "nice", "-n", "10", WHISPER_BIN,
        "-m", WHISPER_MODEL, "-f", str(wav),
        "-l", WHISPER_LANG, "-t", str(WHISPER_THREADS),
        "-nt", "-np",
    ]
    if WHISPER_PROMPT:
        cmd += ["--prompt", WHISPER_PROMPT]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"whisper exited with {result.returncode}")
    return clean_text(" ".join(result.stdout.split()))


def cleanup_old_days():
    if KEEP_DAYS <= 0:
        return
    cutoff = (datetime.date.today() - datetime.timedelta(days=KEEP_DAYS)).isoformat()
    for folder in day_dirs():
        if folder.name < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            log(f"Deleted recordings from {folder.name} (older than {KEEP_DAYS} days)")


def transcribe_clip(meta_path, redo=False):
    wav = meta_path.with_suffix(".wav")
    started = time.time()
    try:
        text = transcribe(wav)
        meta = load_meta(meta_path) or {}
        meta["transcript"] = text
        meta["transcript_model"] = MODEL_TAG
        meta["transcribed_at"] = int(time.time())
        meta.pop("retranscribe_error", None)
        save_meta(meta_path, meta)
        STATE["error"] = None
        log(f"{'Re-transcribed' if redo else 'Transcribed'} {meta_path.parent.name}/{wav.name} "
            f"in {time.time() - started:.1f} s: {text[:60]}")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        meta = load_meta(meta_path) or {}
        if redo:
            # Keep the old transcript; don't try this clip again with this model
            meta["retranscribe_error"] = str(exc)[:300]
            meta["transcript_model"] = MODEL_TAG
        else:
            meta["transcript_error"] = str(exc)[:300]
        save_meta(meta_path, meta)
        STATE["error"] = str(exc)[:300]
        log(f"Transcription failed for {wav.name}: {exc}")


def worker():
    available =Path(WHISPER_BIN).is_file() and Path(WHISPER_MODEL).is_file()
    STATE["transcription"] = "on" if available else "off"
    if not available:
        log(f"Transcription off: whisper.cpp not found ({WHISPER_BIN}, {WHISPER_MODEL})")

    last_cleanup = 0.0
    while True:
        try:
            if time.time() - last_cleanup > 3600:
                cleanup_old_days()
                last_cleanup = time.time()

            new, redo = find_pending()
            STATE["pending"], STATE["redo"] = len(new), len(redo)
            if not available or not (new or redo):
                time.sleep(60 if not available else 5)
                continue

            # Work through a few new clips, or one old one, then rescan so new
            # clips always come first
            for meta_path in new[:5] or redo[:1]:
                transcribe_clip(meta_path, redo=not new)
                key = "redo" if not new else "pending"
                STATE[key] = max(0, STATE[key] - 1)
        except Exception as exc:  # keep the worker alive whatever happens
            log(f"Worker error: {exc}")
            time.sleep(10)


# --- HTTP ---------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "ScannerTimeline/1.0"

    def log_message(self, fmt, *args):
        pass

    def send_body(self, status, body, content_type, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_body(status, body, "application/json; charset=utf-8",
                       {"Cache-Control": "no-store"})

    def not_found(self):
        self.send_json({"error": "not found"}, 404)

    def status_info(self):
        return {
            "transcription": STATE["transcription"],
            "pending": STATE["pending"],
            "retranscribe": STATE["redo"],
            "transcription_error": STATE["error"],
        }

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                self.send_body(200, PAGE.read_bytes(), "text/html; charset=utf-8",
                               {"Cache-Control": "no-cache"})
            elif path == "/api/days":
                self.send_json(dict(self.status_info(),
                                    today=datetime.date.today().isoformat(),
                                    days=list_days()))
            elif path.startswith("/api/day/"):
                day = path[len("/api/day/"):]
                if not DATE_RE.match(day):
                    return self.not_found()
                self.send_json(dict(self.status_info(),
                                    today=datetime.date.today().isoformat(),
                                    date=day, clips=clips_for_day(day)))
            elif path == "/api/search":
                query = parse_qs(url.query).get("q", [""])[0].strip()
                results = search(query) if len(query) >= 2 else []
                self.send_json({"query": query, "results": results})
            elif path.startswith("/audio/"):
                self.serve_audio(path[len("/audio/"):])
            else:
                self.not_found()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve_audio(self, rel):
        parts = rel.split("/")
        if len(parts) != 2 or not DATE_RE.match(parts[0]) or not NAME_RE.match(parts[1]):
            return self.not_found()
        file = ARCHIVE / parts[0] / parts[1]
        if not file.is_file():
            return self.not_found()

        size = file.stat().st_size
        start, end, status = 0, size - 1, 200
        match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if match and size > 0:
            first, last = match.groups()
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            elif last:
                start = max(0, size - int(last))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206

        headers = {"Accept-Ranges": "bytes", "Cache-Control": "max-age=86400"}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        with open(file, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_body(status, body, "audio/wav", headers)


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    log(f"Timeline running on http://0.0.0.0:{PORT}/ (archive: {ARCHIVE})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
