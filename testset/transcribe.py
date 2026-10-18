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
            # Filters that were on while you transcribed (ffmpeg -af syntax), if any
            clip["listen_filter"] = str(body.get("listen_filter") or "")
            # Spoken language, "" = not tagged (benchmark.py then guesses from the transcript)
            clip["lang"] = body.get("lang") if body.get("lang") in ("en", "de") else ""
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
.fgrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:4px 20px; margin-top:6px; }
.fgrid label { margin:6px 0 0; }
.fgrid input[type=range] { width:100%; }
.fgrid .chk { display:flex; gap:6px; align-items:center; color:var(--text); font-size:14px; }
.fgrid.off { opacity:.45; }
code.ff { font-size:12px; word-break:break-all; }
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
    <span>#${c.id}</span><span class="meta">${c.lang ? c.lang.toUpperCase() + " · " : ""}${(c.duration || 0).toFixed(1)} s</span></button>`).join("");
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
    <div class="row"><strong>Listening filters</strong>
      <button class="b" id="f-on" title="Ctrl+B in the text field">Filters off</button>
      <span class="muted" style="margin-left:8px">Presets</span>
      ${Object.keys(PRESETS).map(p => `<button class="b preset" data-p="${p}">${p}</button>`).join("")}
    </div>
    <div class="fgrid" id="fgrid">
      <label>High-pass <b id="v-hp"></b><input type="range" id="f-hp" min="0" max="1000" step="50"></label>
      <label>Low-pass <b id="v-lp"></b><input type="range" id="f-lp" min="1500" max="8000" step="100"></label>
      <label>Noise reduction <b id="v-dn"></b><input type="range" id="f-dn" min="0" max="100" step="10"></label>
      <label>Volume <b id="v-g"></b><input type="range" id="f-g" min="0" max="24" step="1"></label>
      <label class="chk"><input type="checkbox" id="f-comp"> Even out loud and quiet speakers</label>
    </div>
    <p class="muted" style="margin:10px 0 0">About the same in ffmpeg: <code class="ff" id="f-ff"></code> <span id="f-busy"></span></p>
  </div>
  <div class="card">
    <label for="ref">Your transcript</label>
    <textarea id="ref" rows="4" placeholder="Write exactly what is said…"></textarea>
    <label for="notes">Notes (optional: speaker, quality, why it's hard)</label>
    <input type="text" id="notes">
    <label>Language</label>
    <div class="row">
      <button class="b lang" data-l="en">English</button>
      <button class="b lang" data-l="de">German</button>
      <span class="muted">Untagged clips are guessed from the transcript when scoring.</span>
    </div>
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
    <kbd>Ctrl</kbd>+<kbd>[</kbd>/<kbd>]</kbd> −/+2 s · <kbd>Ctrl</kbd>+<kbd>B</kbd> filters on/off ·<kbd>Ctrl</kbd>+<kbd>Enter</kbd> save &amp; next · <kbd>Ctrl</kbd>+<kbd>↑</kbd>/<kbd>↓</kbd> previous/next clip.<br>
    Numbers as digits or words (English or German), NATO spelling or letters: all are normalised before scoring. Write <code>??</code> for a word you can't make out; it is not scored.</p>`;
  const a = $("#audio"), ref = $("#ref");
  ref.value = cur.reference || ""; $("#notes").value = cur.notes || "";
  $("#status").textContent = cur.updated_at ? "Saved " + fmtTime(cur.updated_at) : "";
  const rate = parseFloat(localStorage.getItem("rate") || "1");
  const setRate = s => { a.playbackRate = a.defaultPlaybackRate = s; try { localStorage.setItem("rate", s); } catch (e) {}
    document.querySelectorAll(".speed").forEach(b => b.classList.toggle("on", parseFloat(b.dataset.s) === s)); };
  setRate(rate);
  a.loop = localStorage.getItem("loop") === "1"; $("#loop").classList.toggle("on", a.loop);
  $("#replay").onclick = () => { a.currentTime = 0; a.play(); };
  $("#back").onclick = () => { a.currentTime = Math.max(0, a.currentTime - 2); };
  $("#fwd").onclick = () => { a.currentTime = Math.min(a.duration || 0, a.currentTime + 2); };
  document.querySelectorAll(".speed").forEach(b => b.onclick = () => setRate(parseFloat(b.dataset.s)));
  $("#loop").onclick = () => { a.loop = !a.loop; $("#loop").classList.toggle("on", a.loop); try { localStorage.setItem("loop", a.loop ? "1" : "0"); } catch (e) {} };
  ref.oninput = $("#notes").oninput = () => { dirty = true; $("#status").textContent = "Unsaved changes"; };
  const setLang = l => document.querySelectorAll(".lang").forEach(b => b.classList.toggle("on", b.dataset.l === l));
  setLang(cur.lang || "");
  document.querySelectorAll(".lang").forEach(b => b.onclick = () => {
    setLang(b.classList.contains("on") ? "" : b.dataset.l);  // click again to untag
    ref.oninput();
  });
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
    else if (e.ctrlKey && e.key.toLowerCase() === "b") { e.preventDefault(); $("#f-on").click(); }
  };
  attachFilters(a);
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
    body: JSON.stringify({ reference, notes: notes.value, listen_filter: reference === null ? "" : activeFilterString(),
                           lang: (document.querySelector(".lang.on") || {dataset: {}}).dataset.l || "" }) });
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

// --- Listening filters --------------------------------------------------------
// High-pass, low-pass, compressor and volume run live in Web Audio on the
// <audio> element. Noise reduction is spectral subtraction on the decoded clip;
// the result replaces the element's source, so seeking, speed and loop still work.

const LP_OFF = 8000;
const PRESETS = {
  "Voice band": { hp: 300, lp: 3400, dn: 0, comp: false, g: 0 },
  "+ level": { hp: 300, lp: 3400, dn: 0, comp: true, g: 6 },
  "+ denoise": { hp: 300, lp: 3400, dn: 50, comp: true, g: 6 },
};
const F = { s: loadFilterSettings(), ctx: null, audio: null, src: null, orig: null,
            decoded: null, decodedFor: null, cache: {}, timer: null };

function loadFilterSettings() {
  const base = { on: false, hp: 300, lp: 3400, dn: 0, comp: false, g: 0 };
  try { return Object.assign(base, JSON.parse(localStorage.getItem("filters") || "{}")); } catch (e) { return base; }
}

function attachFilters(a) {
  if (F.src) { try { F.src.disconnect(); } catch (e) {} }
  F.audio = a; F.src = null; F.orig = a.src;
  const s = F.s;
  $("#f-hp").value = s.hp; $("#f-lp").value = s.lp; $("#f-dn").value = s.dn; $("#f-g").value = s.g; $("#f-comp").checked = s.comp;
  const read = () => { s.hp = +$("#f-hp").value; s.lp = +$("#f-lp").value; s.dn = +$("#f-dn").value;
                       s.g = +$("#f-g").value; s.comp = $("#f-comp").checked; s.on = true; applyFilters(); };
  ["#f-hp", "#f-lp", "#f-dn", "#f-g"].forEach(id => $(id).oninput = read);
  $("#f-comp").onchange = read;
  $("#f-on").onclick = () => { s.on = !s.on; applyFilters(); };
  document.querySelectorAll(".preset").forEach(b => b.onclick = () => {
    Object.assign(s, PRESETS[b.dataset.p], { on: true });
    $("#f-hp").value = s.hp; $("#f-lp").value = s.lp; $("#f-dn").value = s.dn; $("#f-g").value = s.g; $("#f-comp").checked = s.comp;
    applyFilters();
  });
  a.addEventListener("play", () => { if (F.ctx) F.ctx.resume(); });
  applyFilters();
}

function applyFilters() {
  const s = F.s;
  try { localStorage.setItem("filters", JSON.stringify(s)); } catch (e) {}
  $("#v-hp").textContent = s.hp ? s.hp + " Hz" : "off";
  $("#v-lp").textContent = s.lp < LP_OFF ? s.lp + " Hz" : "off";
  $("#v-dn").textContent = s.dn ? s.dn + " %" : "off";
  $("#v-g").textContent = "+" + s.g + " dB";
  $("#f-on").textContent = s.on ? "Filters on" : "Filters off";
  $("#f-on").classList.toggle("on", s.on);
  $("#fgrid").classList.toggle("off", !s.on);
  $("#f-ff").textContent = filterString() || "(no filter)";
  document.querySelectorAll(".preset").forEach(b => b.classList.toggle("on", s.on &&
    Object.entries(PRESETS[b.dataset.p]).every(([k, v]) => s[k] === v)));

  const live = s.on && (s.hp > 0 || s.lp < LP_OFF || s.comp || s.g > 0);
  if (live || F.src) wire(live);
  clearTimeout(F.timer);
  F.timer = setTimeout(updateSource, s.on && s.dn ? 250 : 0);
}

function wire(live) {
  if (!F.ctx) F.ctx = new (window.AudioContext || window.webkitAudioContext)();
  const c = F.ctx, s = F.s;
  if (!F.src) {
    F.src = c.createMediaElementSource(F.audio);
    F.hp = c.createBiquadFilter(); F.hp.type = "highpass"; F.hp.Q.value = 0;
    F.lp = c.createBiquadFilter(); F.lp.type = "lowpass"; F.lp.Q.value = 0;
    F.comp = c.createDynamicsCompressor();
    F.comp.threshold.value = -35; F.comp.knee.value = 10; F.comp.ratio.value = 8;
    F.comp.attack.value = 0.003; F.comp.release.value = 0.25;
    F.gain = c.createGain();
  }
  [F.src, F.hp, F.lp, F.comp, F.gain].forEach(n => n.disconnect());
  const chain = [F.src];
  if (live) {
    if (s.hp > 0) { F.hp.frequency.value = s.hp; chain.push(F.hp); }
    if (s.lp < LP_OFF) { F.lp.frequency.value = s.lp; chain.push(F.lp); }
    if (s.comp) chain.push(F.comp);
    F.gain.gain.value = Math.pow(10, s.g / 20); chain.push(F.gain);
  }
  chain.push(c.destination);
  for (let i = 0; i < chain.length - 1; i++) chain[i].connect(chain[i + 1]);
  if (!F.audio.paused) c.resume();
}

// Switches the <audio> element between the original clip and a denoised copy.
async function updateSource() {
  const s = F.s, a = F.audio, clip = cur;
  let url = F.orig;
  if (s.on && s.dn) {
    $("#f-busy").textContent = "· reducing noise…";
    try { url = await denoisedUrl(clip, s.dn); }
    catch (e) { $("#f-busy").textContent = "· noise reduction failed: " + e.message; return; }
    $("#f-busy").textContent = "";
    if (a !== F.audio || cur !== clip || !(F.s.on && F.s.dn === s.dn)) return; // settings moved on meanwhile
  } else {
    $("#f-busy").textContent = "";
  }
  if (a.src === url) return;
  const t = a.currentTime, playing = !a.paused;
  a.src = url;
  a.addEventListener("loadedmetadata", () => {
    a.currentTime = Math.min(t, a.duration || t);
    if (playing) a.play().catch(() => {});
  }, { once: true });
}

async function denoisedUrl(clip, strength) {
  const key = clip.audio + "|" + strength;
  if (F.cache[key]) return F.cache[key];
  if (F.decodedFor !== clip.audio) {
    if (!F.ctx) F.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const bytes = await fetch("/" + clip.audio).then(r => r.arrayBuffer());
    F.decoded = await F.ctx.decodeAudioData(bytes);
    F.decodedFor = clip.audio;
  }
  const out = spectralSubtract(F.decoded.getChannelData(0), strength / 100);
  return F.cache[key] = URL.createObjectURL(wavBlob(out, F.decoded.sampleRate));
}

// Estimates the noise spectrum from the quietest 15 % of frames and subtracts it.
function spectralSubtract(x, k) {
  const N = 1024, H = N / 4, nb = N / 2 + 1;
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const frames = Math.max(1, Math.ceil(Math.max(0, x.length - N) / H) + 1);
  const xp = new Float64Array((frames - 1) * H + N); xp.set(x);
  const re = new Float64Array(N), im = new Float64Array(N);
  const load = f => { for (let i = 0; i < N; i++) { re[i] = xp[f * H + i] * win[i]; im[i] = 0; } fft(re, im, false); };

  const mags = new Float32Array(frames * nb), energy = new Float64Array(frames);
  for (let f = 0; f < frames; f++) {
    load(f);
    for (let b = 0; b < nb; b++) { const m = Math.hypot(re[b], im[b]); mags[f * nb + b] = m; energy[f] += m * m; }
  }
  const order = Array.from(energy.keys()).sort((p, q) => energy[p] - energy[q]);
  const quiet = order.slice(0, Math.max(3, Math.round(frames * 0.15)));
  const noise = new Float64Array(nb);
  for (const f of quiet) for (let b = 0; b < nb; b++) noise[b] += mags[f * nb + b] / quiet.length;

  const alpha = 1 + 3 * k;                        // over-subtraction
  const floor = Math.pow(10, -(6 + 24 * k) / 20); // never cut a bin by more than 6..30 dB
  const gain = new Float64Array(nb).fill(1);
  const y = new Float64Array(xp.length);
  for (let f = 0; f < frames; f++) {
    load(f);
    for (let b = 0; b < nb; b++) {
      const m = mags[f * nb + b] || 1e-12, ratio = noise[b] / m;
      const g = Math.sqrt(Math.max(1 - alpha * ratio * ratio, floor * floor));
      gain[b] = 0.5 * gain[b] + 0.5 * g;          // smooth over time against "musical" noise
      re[b] *= gain[b]; im[b] *= gain[b];
      if (b > 0 && b < N / 2) { re[N - b] *= gain[b]; im[N - b] *= gain[b]; }
    }
    fft(re, im, true);
    for (let i = 0; i < N; i++) y[f * H + i] += re[i] * win[i] / 1.5; // Hann² at 75 % overlap sums to 1.5
  }
  return y.subarray(0, x.length);
}

function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = 2 * Math.PI / len * (inverse ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < half; j++) {
        const p = i + j, q = p + half;
        const tr = re[q] * cr - im[q] * ci, ti = re[q] * ci + im[q] * cr;
        re[q] = re[p] - tr; im[q] = im[p] - ti; re[p] += tr; im[p] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function wavBlob(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

// The current settings in ffmpeg -af syntax (approximate: ffmpeg's afftdn
// and acompressor are not the same algorithms as the browser's).
function filterString() {
  const s = F.s, parts = [];
  if (s.hp > 0) parts.push(`highpass=f=${s.hp}`);
  if (s.lp < LP_OFF) parts.push(`lowpass=f=${s.lp}`);
  if (s.dn) parts.push(`afftdn=nr=${Math.round(6 + 24 * s.dn / 100)}`);
  if (s.comp) parts.push("acompressor=threshold=-35dB:ratio=8:attack=3:release=250");
  if (s.g > 0) parts.push(`volume=${s.g}dB`);
  return parts.join(",");
}

function activeFilterString() { return F.s.on ? filterString() : ""; }

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
