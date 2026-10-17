#!/usr/bin/env python3
"""
Builds a Whisper test set from the recorder's archive.

On the first run it picks COUNT clips from the archive, copies them into
DATA/clips/ and writes DATA/testset.json. It then serves a page where each clip
can be replayed and transcribed by hand; the transcripts are saved into
testset.json next to the clip they belong to. benchmark.py scores Whisper
against them.

Uses only the Python standard library.

    python3 transcribe.py                 # select 20 clips (first run) and serve
    python3 transcribe.py --count 30      # select more if the set has fewer
"""
import argparse
import collections
import json
import os
import random
import re
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
HOME = Path.home()
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CLIP_RE = re.compile(r"^[A-Za-z0-9_.-]+\.wav$")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# --- Test set file ------------------------------------------------------------

class TestSet:
    def __init__(self, data_dir, archive):
        self.data = Path(data_dir)
        self.archive = Path(archive)
        self.clips_dir = self.data / "clips"
        self.path = self.data / "testset.json"
        self.doc = self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"created": int(time.time()), "clips": []}

    def save(self):
        self.data.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    @property
    def clips(self):
        return self.doc["clips"]

    def get(self, clip_id):
        return next((c for c in self.clips if c["id"] == clip_id), None)

    # --- Selecting clips from the archive ---

    def candidates(self, min_dur, max_dur):
        used = {c["source"] for c in self.clips} | set(self.doc.get("rejected", []))
        by_day = collections.defaultdict(list)
        if not self.archive.is_dir():
            return by_day
        for day in sorted(self.archive.iterdir()):
            if not (day.is_dir() and DATE_RE.match(day.name)):
                continue
            for meta_path in day.glob("*.json"):
                wav = meta_path.with_suffix(".wav")
                source = f"{day.name}/{wav.name}"
                if source in used or not wav.exists():
                    continue
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except (OSError, ValueError):
                    continue
                duration = meta.get("duration") or 0
                if meta.get("transcript_error") or not (min_dur <= duration <= max_dur):
                    continue
                by_day[day.name].append((source, wav, meta))
        return by_day

    def pick(self, n, min_dur, max_dur, rng):
        """Picks n clips, spread round-robin over the days in the archive."""
        by_day = self.candidates(min_dur, max_dur)
        for clips in by_day.values():
            rng.shuffle(clips)
        days = list(by_day)
        rng.shuffle(days)
        picked = []
        while len(picked) < n and days:
            for day in list(days):
                if not by_day[day]:
                    days.remove(day)
                    continue
                picked.append(by_day[day].pop())
                if len(picked) >= n:
                    break
        return picked

    def make_entry(self, clip_id, source, wav, meta):
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        audio = f"{clip_id}_{source.replace('/', '_')}"
        shutil.copy2(wav, self.clips_dir / audio)
        return {
            "id": clip_id,
            "audio": f"clips/{audio}",
            "source": source,
            "start": meta.get("start"),
            "duration": meta.get("duration"),
            "frequency": meta.get("frequency"),
            # What the live system produced; shown only on request so it
            # doesn't bias the hand transcript.
            "production_transcript": meta.get("transcript"),
            "reference": None,
            "notes": "",
            "updated_at": None,
        }

    def fill(self, count, min_dur, max_dur, seed):
        missing = count - len(self.clips)
        if missing <= 0:
            return 0
        rng = random.Random(seed)
        next_num = max([int(c["id"]) for c in self.clips] + [0]) + 1
        for source, wav, meta in self.pick(missing, min_dur, max_dur, rng):
            self.clips.append(self.make_entry(f"{next_num:02d}", source, wav, meta))
            next_num += 1
        self.save()
        return len(self.clips) - (count - missing)

    def replace(self, clip_id, min_dur, max_dur):
        """Swaps a clip for a fresh random one; the old one is never picked again."""
        old = self.get(clip_id)
        if old is None:
            return None
        picked = self.pick(1, min_dur, max_dur, random.Random())
        if not picked:
            return None
        self.doc.setdefault("rejected", []).append(old["source"])
        try:
            (self.data / old["audio"]).unlink()
        except OSError:
            pass
        new = self.make_entry(clip_id, *picked[0])
        self.clips[self.clips.index(old)] = new
        self.save()
        return new


# --- HTTP ---------------------------------------------------------------------

def make_handler(testset, opts):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TestSet/1.0"

        def log_message(self, fmt, *args):
            pass

        def send_body(self, status, body, content_type, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, obj, status=200):
            self.send_body(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

        def not_found(self):
            self.send_json({"error": "not found"}, 404)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path in ("/", "/index.html"):
                    self.send_body(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/clips":
                    self.send_json({"clips": testset.clips,
                                    "report": (testset.data / "results/report.html").is_file()})
                elif path == "/report":
                    report = testset.data / "results/report.html"
                    if not report.is_file():
                        return self.not_found()
                    self.send_body(200, report.read_bytes(), "text/html; charset=utf-8")
                elif path.startswith("/clips/"):
                    self.serve_audio(path[len("/clips/"):])
                else:
                    self.not_found()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self.send_json({"error": "bad json"}, 400)
            m = re.match(r"^/api/clip/(\d+)(/replace)?$", path)
            if not m:
                return self.not_found()
            clip = testset.get(m.group(1))
            if clip is None:
                return self.not_found()
            if m.group(2):
                new = testset.replace(clip["id"], opts.min_duration, opts.max_duration)
                if new is None:
                    return self.send_json({"error": "no unused clips left in the archive"}, 409)
                log(f"Replaced clip {clip['id']}: {clip['source']} -> {new['source']}")
                return self.send_json(new)
            ref = body.get("reference")
            clip["reference"] = None if ref is None else " ".join(str(ref).split())
            clip["notes"] = str(body.get("notes") or "").strip()
            clip["updated_at"] = int(time.time())
            testset.save()
            self.send_json(clip)

        def serve_audio(self, name):
            if not CLIP_RE.match(name):
                return self.not_found()
            file = testset.clips_dir / name
            if not file.is_file():
                return self.not_found()
            size = file.stat().st_size
            start, end, status = 0, size - 1, 200
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            if m and size > 0:
                first, last = m.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                elif last:
                    start = max(0, size - int(last))
                status = 206
            extra = {"Accept-Ranges": "bytes"}
            if status == 206:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            with open(file, "rb") as f:
                f.seek(start)
                self.send_body(status, f.read(end - start + 1), "audio/wav", extra)

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whisper test set</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --text:#1d232b; --muted:#68707c; --line:#dde1e6;
        --accent:#2563eb; --ok:#15803d; --warn:#b45309; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --panel:#1d2128; --text:#e4e7eb; --muted:#9aa3ae; --line:#2f353f;
          --accent:#60a5fa; --ok:#4ade80; --warn:#fbbf24; } }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--text); }
header { padding:12px 20px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:17px; margin:0; }
header .progress { color:var(--muted); }
header a { color:var(--accent); margin-left:auto; }
main { display:grid; grid-template-columns:260px 1fr; min-height:calc(100vh - 50px); }
@media (max-width:760px) { main { grid-template-columns:1fr; } nav { max-height:30vh; } }
nav { border-right:1px solid var(--line); overflow:auto; }
nav button { display:flex; gap:8px; width:100%; padding:8px 14px; border:0; border-bottom:1px solid var(--line);
  background:none; color:inherit; font:inherit; text-align:left; cursor:pointer; }
nav button.active { background:var(--panel); box-shadow:inset 3px 0 var(--accent); }
nav .mark { width:16px; color:var(--ok); }
nav .mark.empty { color:var(--warn); }
nav .meta { color:var(--muted); font-size:13px; margin-left:auto; }
section { padding:20px; max-width:900px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-bottom:16px; }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
audio { width:100%; margin:10px 0; }
button.b { padding:6px 12px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:inherit; font:inherit; cursor:pointer; }
button.b.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button.b.on { border-color:var(--accent); color:var(--accent); }
textarea { width:100%; font:16px/1.5 ui-monospace, Consolas, monospace; padding:10px; border:1px solid var(--line);
  border-radius:6px; background:var(--bg); color:inherit; resize:vertical; }
input[type=text] { width:100%; padding:6px 10px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:inherit; font:inherit; }
label { font-size:13px; color:var(--muted); display:block; margin:12px 0 4px; }
.muted { color:var(--muted); font-size:13px; }
.status { min-height:1.4em; font-size:13px; color:var(--muted); }
.prod { font-family:ui-monospace, Consolas, monospace; padding:8px; border:1px dashed var(--line); border-radius:6px; margin-top:8px; }
kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; font-size:12px; }
details { margin-top:8px; } summary { cursor:pointer; color:var(--muted); font-size:13px; }
</style></head>
<body>
<header><h1>Whisper test set</h1><span class="progress" id="progress"></span><a id="report" href="/report" target="_blank" hidden>Benchmark report</a></header>
<main>
<nav id="list"></nav>
<section id="detail"><p class="muted">Loading…</p></section>
</main>
<script>
let clips = [], cur = null, dirty = false;
const $ = s => document.querySelector(s);
const fmtTime = t => t ? new Date(t * 1000).toLocaleString() : "";

async function load() {
  const r = await fetch("/api/clips").then(r => r.json());
  clips = r.clips; $("#report").hidden = !r.report;
  renderList();
  const first = clips.find(c => c.reference === null) || clips[0];
  if (first) select(first.id); else $("#detail").innerHTML = "<p>No clips in the test set. Is the archive path right?</p>";
}

function renderList() {
  const done = clips.filter(c => c.reference !== null).length;
  $("#progress").textContent = `${done} of ${clips.length} transcribed`;
  $("#list").innerHTML = clips.map(c => `<button data-id="${c.id}" class="${cur && cur.id === c.id ? "active" : ""}">
    <span class="mark ${c.reference === "" ? "empty" : ""}">${c.reference === null ? "" : c.reference === "" ? "∅" : "✓"}</span>
    <span>#${c.id}</span><span class="meta">${(c.duration || 0).toFixed(1)} s</span></button>`).join("");
  $("#list").querySelectorAll("button").forEach(b => b.onclick = () => select(b.dataset.id));
}

async function select(id) {
  if (dirty && cur) await save(false);
  cur = clips.find(c => c.id === id); dirty = false;
  renderList();
  $("#detail").innerHTML = `
  <div class="card">
    <div class="row"><strong>Clip #${cur.id}</strong><span class="muted">${cur.source} · ${fmtTime(cur.start)} · ${(cur.duration || 0).toFixed(1)} s</span></div>
    <audio id="audio" controls preload="auto" src="/${cur.audio}"></audio>
    <div class="row">
      <button class="b" id="replay">⟲ From start</button>
      <button class="b" id="back">−2 s</button>
      <button class="b" id="fwd">+2 s</button>
      <span class="muted" style="margin-left:8px">Speed</span>
      ${[0.6, 0.75, 1, 1.25].map(s => `<button class="b speed" data-s="${s}">${s}×</button>`).join("")}
      <button class="b" id="loop">Loop</button>
    </div>
  </div>
  <div class="card">
    <label for="ref">Your transcript</label>
    <textarea id="ref" rows="4" placeholder="Write exactly what is said…"></textarea>
    <label for="notes">Notes (optional: speaker, quality, why it's hard)</label>
    <input type="text" id="notes">
    <div class="row" style="margin-top:12px">
      <button class="b primary" id="savenext">Save &amp; next</button>
      <button class="b" id="save">Save</button>
      <button class="b" id="silent" title="Saves an empty transcript: Whisper should output nothing here">Only noise / no speech</button>
      <button class="b" id="replace" title="Swap this clip for another random one">Replace clip</button>
      <button class="b" id="clear" title="Mark as not transcribed yet">Clear</button>
    </div>
    <div class="status" id="status"></div>
    <details><summary>Show what the live Whisper produced (look after you've written yours)</summary>
      <div class="prod">${cur.production_transcript == null ? "<em>not transcribed</em>" : escapeHtml(cur.production_transcript) || "<em>empty</em>"}</div></details>
  </div>
  <p class="muted">Keys in the text field: <kbd>Ctrl</kbd>+<kbd>Space</kbd> play/pause · <kbd>Esc</kbd> replay from start ·
    <kbd>Ctrl</kbd>+<kbd>[</kbd>/<kbd>]</kbd> −/+2 s · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save &amp; next · <kbd>Ctrl</kbd>+<kbd>↑</kbd>/<kbd>↓</kbd> previous/next clip.<br>
    Numbers as digits or words, NATO spelling or letters: both are normalised before scoring. Write <code>??</code> for a word you can't make out; it is not scored.</p>`;
  const a = $("#audio"), ref = $("#ref");
  ref.value = cur.reference || ""; $("#notes").value = cur.notes || "";
  $("#status").textContent = cur.updated_at ? "Saved " + fmtTime(cur.updated_at) : "";
  const rate = parseFloat(localStorage.getItem("rate") || "1");
  const setRate = s => { a.playbackRate = s; try { localStorage.setItem("rate", s); } catch (e) {}
    document.querySelectorAll(".speed").forEach(b => b.classList.toggle("on", parseFloat(b.dataset.s) === s)); };
  setRate(rate);
  a.loop = localStorage.getItem("loop") === "1"; $("#loop").classList.toggle("on", a.loop);
  $("#replay").onclick = () => { a.currentTime = 0; a.play(); };
  $("#back").onclick = () => { a.currentTime = Math.max(0, a.currentTime - 2); };
  $("#fwd").onclick = () => { a.currentTime = Math.min(a.duration || 0, a.currentTime + 2); };
  document.querySelectorAll(".speed").forEach(b => b.onclick = () => setRate(parseFloat(b.dataset.s)));
  $("#loop").onclick = () => { a.loop = !a.loop; $("#loop").classList.toggle("on", a.loop); try { localStorage.setItem("loop", a.loop ? "1" : "0"); } catch (e) {} };
  ref.oninput = $("#notes").oninput = () => { dirty = true; $("#status").textContent = "Unsaved changes"; };
  $("#save").onclick = () => save(false);
  $("#savenext").onclick = () => save(true);
  $("#silent").onclick = () => { ref.value = ""; save(true, ""); };
  $("#clear").onclick = () => save(false, null);
  $("#replace").onclick = replace;
  ref.onkeydown = e => {
    if (e.ctrlKey && e.code === "Space") { e.preventDefault(); a.paused ? a.play() : a.pause(); }
    else if (e.key === "Escape") { e.preventDefault(); a.currentTime = 0; a.play(); }
    else if (e.ctrlKey && e.key === "[") { e.preventDefault(); $("#back").click(); }
    else if (e.ctrlKey && e.key === "]") { e.preventDefault(); $("#fwd").click(); }
    else if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); save(true); }
    else if (e.ctrlKey && e.key === "ArrowDown") { e.preventDefault(); step(1); }
    else if (e.ctrlKey && e.key === "ArrowUp") { e.preventDefault(); step(-1); }
  };
  ref.focus();
  a.play().catch(() => {});
}

function step(d) {
  const i = clips.findIndex(c => c.id === cur.id) + d;
  if (i >= 0 && i < clips.length) select(clips[i].id);
}

async function save(next, override) {
  const ref = $("#ref"), notes = $("#notes");
  const reference = override !== undefined ? override : ref.value.trim();
  if (override === undefined && reference === "" && !confirm("Empty transcript means 'no speech in this clip'. Save it like that?")) return;
  const r = await fetch(`/api/clip/${cur.id}`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reference, notes: notes.value }) });
  if (!r.ok) { $("#status").textContent = "Save failed: " + r.status; return; }
  Object.assign(cur, await r.json()); dirty = false;
  $("#status").textContent = "Saved " + fmtTime(cur.updated_at);
  renderList();
  if (next) {
    const i = clips.findIndex(c => c.id === cur.id);
    const nxt = clips.slice(i + 1).find(c => c.reference === null) || clips.find(c => c.reference === null);
    if (nxt) select(nxt.id); else $("#status").textContent = "All clips transcribed. Run benchmark.py next.";
  }
}

async function replace() {
  if (!confirm("Swap this clip for another random one from the archive?")) return;
  const r = await fetch(`/api/clip/${cur.id}/replace`, { method: "POST", body: "{}" });
  if (!r.ok) { $("#status").textContent = (await r.json()).error || "Replace failed"; return; }
  const fresh = await r.json();
  clips[clips.findIndex(c => c.id === cur.id)] = fresh; dirty = false;
  select(fresh.id);
}

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }
window.addEventListener("beforeunload", e => { if (dirty) e.preventDefault(); });
load();
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", default=os.environ.get("ARCHIVE", str(HOME / "scanner-archive")),
                   help="recorder archive with YYYY-MM-DD folders (default: $ARCHIVE)")
    p.add_argument("--data", default=str(HERE / "data"), help="where the test set is kept")
    p.add_argument("--count", type=int, default=20, help="clips in the test set (default 20)")
    p.add_argument("--min-duration", type=float, default=1.5, help="skip shorter clips (s)")
    p.add_argument("--max-duration", type=float, default=30.0, help="skip longer clips (s)")
    p.add_argument("--seed", type=int, default=None, help="random seed for the selection")
    p.add_argument("--port", type=int, default=8091)
    opts = p.parse_args()

    testset = TestSet(opts.data, opts.archive)
    added = testset.fill(opts.count, opts.min_duration, opts.max_duration, opts.seed)
    if added:
        log(f"Selected {added} new clips from {opts.archive}")
    if len(testset.clips) < opts.count:
        log(f"Only {len(testset.clips)} suitable clips found in {opts.archive}")
    done = sum(c["reference"] is not None for c in testset.clips)
    log(f"Test set: {testset.path} ({done}/{len(testset.clips)} transcribed)")

    server = ThreadingHTTPServer(("0.0.0.0", opts.port), make_handler(testset, opts))
    server.daemon_threads = True
    log(f"Open http://localhost:{opts.port}/ (or http://adsb.local:{opts.port}/)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
