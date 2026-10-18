"""
Prompt lab for the test set page (transcribe.py).

Runs a configuration from configs.json with a different prompt over the
transcribed clips and scores it against the same configuration with its usual
prompt. Uses benchmark.py's runner, cache and scoring, so results are shared
with the benchmark. Also suggests prompt terms extracted from transcripts.

Uses only the Python standard library.
"""
import collections
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import benchmark


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


class PromptLab:
    def __init__(self, testset, configs_path):
        self.testset = testset
        self.configs_path = Path(configs_path)
        self.history_path = testset.data / "results" / "prompt-lab.json"
        self.lock = threading.Lock()
        self.status = {"state": "idle"}          # idle, running, done or error
        self.opts = SimpleNamespace(
            whisper_bin=benchmark.DEFAULT_BIN,
            threads=os.environ.get("WHISPER_THREADS") or str(os.cpu_count() or 4),
            nice=10, force=False)
        self._archive_terms = None               # (time, suggestions), see suggest()

    # --- Configurations and history ---

    def configs(self):
        try:
            return json.loads(self.configs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def config(self, name):
        return next((c for c in self.configs() if c["name"] == name), None)

    def history(self):
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _save_history(self, entries):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")

    def clips(self):
        return [c for c in self.testset.clips if c.get("reference") is not None]

    @staticmethod
    def variant(base, prompt):
        """The base configuration with another prompt ("" = no prompt)."""
        cfg = dict(base)
        cfg["name"] = f"lab {base['name']} {hashlib.sha1(prompt.encode()).hexdigest()[:8]}"
        cfg["prompt"] = prompt if prompt.strip() else False
        return cfg

    # --- Running ---

    def start(self, base_name, prompt):
        base = self.config(base_name)
        if base is None:
            return "unknown configuration"
        if not self.clips():
            return "no transcribed clips yet"
        with self.lock:
            if self.status["state"] == "running":
                return "busy"
            self.status = {"state": "running", "base": base_name, "phase": "starting",
                           "done": 0, "total": 0, "started": int(time.time())}
        threading.Thread(target=self.run, args=(base, prompt), daemon=True).start()
        return "started"

    def _progress(self, done, total):
        self.status.update(done=done, total=total)

    def run(self, base, prompt):
        clips = self.clips()
        variant = self.variant(base, prompt)
        try:
            for phase, cfg in (("usual prompt", base), ("your prompt", variant)):
                self.status.update(phase=phase, done=0, total=0)
                benchmark.run_config(cfg, clips, self.testset.data, self.opts, progress=self._progress)
            result = self.result(variant["name"], base["name"], prompt)
            entry = {"name": variant["name"], "base": base["name"], "prompt": prompt,
                     "wer": result["lab"]["wer"], "base_wer": result["base"]["wer"],
                     "cer": result["lab"]["cer"], "noise": result["lab"]["noise"],
                     "clips": result["lab"]["clips"], "at": int(time.time())}
            entries = [e for e in self.history() if e["name"] != entry["name"]] + [entry]
            self._save_history(entries)
            self.status = {"state": "done", "name": variant["name"]}
            log(f"Prompt lab: {variant['name']} WER {entry['wer']:.1%} (usual prompt {entry['base_wer']:.1%})")
        except Exception as exc:   # report any failure on the page
            self.status = {"state": "error", "error": str(exc)[:300]}
            log(f"Prompt lab failed: {exc}")

    # --- Results ---

    def _cached(self, cfg):
        path = self.testset.data / "results" / "runs" / f"{benchmark.safe_name(cfg['name'])}.json"
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return cache["clips"] if cache.get("key") == benchmark.config_key(cfg) else {}

    @staticmethod
    def _summary(r):
        return {"wer": r["wer"], "cer": r["cer"], "sub": r["sub"], "del": r["del"], "ins": r["ins"],
                "exact": r["exact"], "clips": r["clips"], "noise": f"{r['silent_halluc']}/{r['silent']}",
                "rtf": r["rtf"]}

    def result(self, name, base_name=None, prompt=None):
        """Scores a lab run (by history name, or base name + prompt) from the cache."""
        if base_name is None:
            entry = next((e for e in self.history() if e["name"] == name), None)
            if entry is None:
                return None
            base_name, prompt = entry["base"], entry["prompt"]
        base = self.config(base_name)
        if base is None:
            return None
        variant = self.variant(base, prompt)
        clips = self.clips()
        b_out, l_out = self._cached(base), self._cached(variant)
        both = [c for c in clips if c["id"] in b_out and c["id"] in l_out]
        b = benchmark.score(base["name"], b_out, both, True)
        l = benchmark.score(variant["name"], l_out, both, True)
        per_clip = []
        for c in both:
            bp, lp = b["per_clip"][c["id"]], l["per_clip"][c["id"]]
            per_clip.append({"id": c["id"], "reference": c["reference"], "lang": benchmark.clip_language(c),
                             "base_errors": bp["errors"], "lab_errors": lp["errors"], "words": lp["words"],
                             "base_html": benchmark.render_ops(bp["ops"]),
                             "lab_html": benchmark.render_ops(lp["ops"])})
        return {"name": variant["name"], "base_name": base["name"], "prompt": prompt,
                "base_prompt": benchmark.prompt_of(base), "missing": len(clips) - len(both),
                "base": self._summary(b), "lab": self._summary(l), "per_clip": per_clip}

    # --- Keeping a prompt ---

    def save_config(self, name, base_name, prompt):
        """Adds the lab prompt to configs.json as a regular benchmark configuration."""
        name = " ".join(str(name).split())
        base = self.config(base_name)
        if not name or base is None:
            return "name and base configuration needed"
        if self.config(name):
            return "a configuration with that name exists already"
        entry = dict(base, name=name, prompt=prompt if prompt.strip() else False)
        text = self.configs_path.read_text(encoding="utf-8").rstrip()
        if not text.endswith("]"):
            return "configs.json doesn't end with ]"
        # Append as one line, keeping the file's hand-made layout
        text = text[:-1].rstrip() + ",\n  " + json.dumps(entry, ensure_ascii=False) + "\n]\n"
        json.loads(text)   # never write a broken file
        self.configs_path.write_text(text, encoding="utf-8")
        return "saved"

    # --- Prompt term suggestions ---

    def suggest(self, source):
        if source == "archive":
            if self._archive_terms and time.time() - self._archive_terms[0] < 600:
                return self._archive_terms[1]
            texts = list(archive_transcripts(self.testset.archive))
            result = extract_terms(texts, min_count=3)
            result["texts"] = len(texts)
            self._archive_terms = (time.time(), result)
            return result
        texts = [c["reference"] for c in self.clips() if c.get("reference")]
        result = extract_terms(texts, min_count=2)
        result["texts"] = len(texts)
        return result


def archive_transcripts(archive):
    archive = Path(archive)
    if not archive.is_dir():
        return
    for day in archive.iterdir():
        if not day.is_dir():
            continue
        for meta_path in day.glob("*.json"):
            try:
                text = json.loads(meta_path.read_text(encoding="utf-8")).get("transcript")
            except (OSError, ValueError):
                continue
            if text:
                yield text


# --- Term extraction -------------------------------------------------------------------
# Heuristics over transcripts: what recurs and looks like a name, a callsign or a
# phrase is a candidate for the prompt. Noise from Whisper's mistakes recurs
# less often, so counts help to tell them apart.

NATO_WORDS = {w: l.upper() for l, w in benchmark.NATO.items()}
NATO_WORDS.update({"alfa": "A", "juliet": "J", "x-ray": "X", "whisky": "W"})
COMMON = set("""
a an the and or but to of in on at for from with by via is are be been it its this that these those
i you he she we they me my your our their his her them us yes no not okay ok roger wilco hello hi
good morning afternoon evening day night thank thanks bye please say again confirm affirmative
negative standby correct over out copy say number
runway tower ground radar approach information info departure apron delivery centre center control
qnh qfe wind squawk transponder report reporting cleared clear contact hold holding short line up
taxi taxiway vacate final downwind base crosswind circuit left right climb climbing descend descending
flight level heading direct ready request traffic visual point feet ft knots degrees gear down
maintain continue expect join joining overhead leaving entering crossing passing turn turning land
landing takeoff take off go around frequency monitor when able time minutes miles north south east
west altitude altimeter vfr ifr ils rnav field airfield airport aircraft
piste rollhalt platzrunde gegenanflug queranflug endanflug anflug abflug landung start frei
freigegeben rollen sie ich wir ihr bitte danke verstanden wiederholen melden zum zur bei über nach mit
fuß grad knoten hektopascal guten morgen tag abend tschüss und der die das den dem auf ist links rechts
null eins zwo zwei drei vier fünf sechs sieben acht neun komma
zero one two three four five six seven eight nine niner decimal hundred thousand
""".split()) | set(NATO_WORDS)


def extract_terms(texts, min_count=2, top=25):
    regs, airlines, names, runways, freqs, phrases = (collections.Counter() for _ in range(6))
    examples = {}
    # Words seen in lower case are ordinary words, even when capitalised at a sentence start
    lowercase_seen = {w for text in texts for w in re.findall(r"\b[a-zäöüß]+\b", text)}
    for text in texts:
        # Registrations written as D-EABC, or spelled out in NATO words
        for m in re.finditer(r"\b([A-Z]{1,2})-([A-Z]{2,4})\b", text):
            regs[m.group(0)] += 1
        words = re.findall(r"[A-Za-z-]+", text.lower())
        run = []
        for w in words + [""]:
            if w in NATO_WORDS:
                run.append(NATO_WORDS[w])
                continue
            if len(run) >= 4:
                regs[f"{run[0]}-{''.join(run[1:])}"] += 1
            run = []
        # Airline-style callsigns: a name followed by a flight number ("Lufthansa 4AB")
        for m in re.finditer(r"\b([A-Z][a-zA-Z]{2,}(?: [A-Z][a-zA-Z]{2,})?) (\d{1,4}[A-Z]{0,2})\b", text):
            name = m.group(1)
            if name.lower().split()[-1] not in COMMON:
                airlines[name] += 1
                examples.setdefault(name, m.group(0))
        # Runways and frequencies
        for m in re.finditer(r"\b(?:runway|piste|rwy)\s+(\d{1,2}\s?[LRC]?)\b", text, re.I):
            runways["runway " + m.group(1).replace(" ", "").upper()] += 1
        for m in re.finditer(r"\b(1[1-3]\d[.,]\d{1,3})\b", text):
            freqs[m.group(1).replace(",", ".")] += 1
        # Capitalised words: names of places, towers, reporting points. At a sentence
        # start only if the word never appears in lower case (radio calls often
        # begin with the station name)
        for m in re.finditer(r"\b([A-ZÄÖÜ][a-zäöüß]{2,}(?:-[A-ZÄÖÜ][a-zäöüß]+)?)\b", text):
            word = m.group(1)
            if word.lower() in COMMON:
                continue
            before = text[:m.start()].rstrip()
            if (not before or before[-1] in ".!?") and word.lower() in lowercase_seen:
                continue
            names[word] += 1
        # Frequent word sequences (2-3 words) within one part of a call (between
        # commas), without numbers, not starting or ending with filler words
        for part in re.split(r"[,.;:!?]", text):
            tokens = [w.lower() for w in re.findall(r"[\wÄÖÜäöüß'-]+", part)]
            for n in (2, 3):
                for i in range(len(tokens) - n + 1):
                    gram = tokens[i:i + n]
                    if (gram[0] in FILLER or gram[-1] in FILLER or any(t in NATO_WORDS for t in gram)
                            or any(ch.isdigit() for t in gram for ch in t)):
                        continue
                    phrases[" ".join(gram)] += 1

    def top_of(counter, limit=top):
        return [{"text": t, "count": n} for t, n in counter.most_common() if n >= min_count][:limit]

    names_list = [x for x in top_of(names, top * 2) if x["text"] not in airlines][:top]
    return {
        "registrations": top_of(regs),
        "callsigns": [dict(x, example=examples.get(x["text"])) for x in top_of(airlines)],
        "names": names_list,
        "runways": top_of(runways, 8),
        "frequencies": top_of(freqs, 10),
        "phrases": _drop_contained(top_of(phrases, top * 2))[:top],
    }


FILLER = set("a an the and or to of in on at for with is are be it this that i you we".split())


def _drop_contained(items):
    """Drops a 2-word phrase when a 3-word phrase containing it is about as frequent."""
    keep = []
    for x in items:
        if any(x["text"] in y["text"] and y["text"] != x["text"] and y["count"] >= x["count"] * 0.8
               for y in items):
            continue
        keep.append(x)
    return keep
