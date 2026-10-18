#!/usr/bin/env python3
"""
Scores whisper.cpp models and settings against the hand transcripts made with
transcribe.py.

Each configuration in configs.json is run over every transcribed clip. Results
are cached per clip in DATA/results/runs/, so adding a configuration only runs
the new one. Prints a ranking and writes DATA/results/summary.csv and
DATA/results/report.html (side-by-side transcripts with the errors marked).

Meant to run in this folder's Docker image (see docker-compose.yml), which has
whisper.cpp and its model converter; the live recorder's install is not used.
Models are kept in ./models/. Missing ones are downloaded from Hugging Face;
fine-tunes in transformers format, such as the ATC models, are converted to
ggml first.

Uses only the Python standard library (the converter needs PyTorch).

    docker compose run --rm benchmark                      # all configurations
    docker compose run --rm benchmark --only tiny,small    # just these (by name)
    docker compose run --rm benchmark --list               # show configurations
"""
import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
# Where the Dockerfile puts whisper.cpp's tools; override to run elsewhere
QUANTIZE_BIN = Path(os.environ.get("WHISPER_QUANTIZE", "/usr/local/bin/whisper-quantize"))
CONVERTER = Path(os.environ.get("WHISPER_CONVERTER", "/opt/convert-h5-to-ggml.py"))
OPENAI_WHISPER = Path(os.environ.get("OPENAI_WHISPER", "/opt/openai-whisper"))  # has whisper/assets/mel_filters.npz
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"
HF = "https://huggingface.co"

# The live system's settings (timeline/server.py)
DEFAULT_PROMPT = ("Air traffic control radio. Runway, taxi, holding point, cleared, QNH, "
                  "squawk, wind, contact tower, report, final, downwind.")
# The same idea for German radio traffic (configs: "prompt": "de")
GERMAN_PROMPT = ("Flugfunk. Piste, Rollhalt, rollen Sie, Platzrunde, Gegenanflug, Queranflug, "
                 "Endanflug, QNH, Transponder, Wind, Landung frei, melden Sie, verstanden.")
DEFAULT_BIN = os.environ.get("WHISPER_BIN", "/usr/local/bin/whisper-cli")

# --- Same clean-up the timeline applies (timeline/server.py, clean_text) ------

NOISE_TAGS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
HALLUCINATIONS = {
    "", "you", "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "thanks for watching.", "bye.", "bye", ".",
}


def clean_text(text, prompt):
    text = NOISE_TAGS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in HALLUCINATIONS:
        return ""
    if prompt and text.strip(" .").lower() in prompt.lower():
        return ""
    return text


# --- Normalisation before scoring --------------------------------------------
# Radio speech is written many ways ("QNH 1013" / "QNH one zero one three" /
# "QNH eins null eins drei", "D-EABC" / "Delta Echo Alpha Bravo Charlie"), so
# both sides are brought into one spoken form: digits as (English) words,
# letters inside callsigns as NATO words. German digits map to the same words,
# so "zwo" and "2" and "two" all match.

DIGITS = "zero one two three four five six seven eight nine".split()
NATO = dict(zip("abcdefghijklmnopqrstuvwxyz", (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()))
SPOKEN = {
    "niner": "nine", "tree": "three", "fower": "four", "fife": "five", "alfa": "alpha",
    "juliet": "juliett", "x-ray": "xray", "whisky": "whiskey", "point": "decimal",
    "okay": "ok", "o.k.": "ok",
    # German
    "null": "zero", "eins": "one", "zwo": "two", "zwei": "two", "drei": "three", "vier": "four",
    "fünf": "five", "fuenf": "five", "sechs": "six", "sieben": "seven", "acht": "eight",
    "neun": "nine", "komma": "decimal", "dezimal": "decimal",
}

# Words that mark a transcript as German, for clips without a language tag
GERMAN_HINTS = set(
    "und der die das den dem sie ihr ich wir bitte danke verstanden wiederholen melden rollen "
    "piste rollhalt platzrunde gegenanflug queranflug endanflug anflug abflug landung start "
    "frei freigegeben zum zur bei über nach mit fuß grad knoten hektopascal guten morgen tag "
    "abend tschüss null eins zwo zwei drei vier fünf sechs sieben acht neun komma".split())


def guess_language(text):
    words = re.findall(r"\w+", (text or "").lower())
    if not words:
        return "none"
    return "de" if sum(w in GERMAN_HINTS for w in words) >= 2 else "en"


def clip_language(clip):
    """The language tag from the transcription page, or a guess from the transcript."""
    return clip.get("lang") or guess_language(clip.get("reference"))


def normalize(text, atc=True):
    t = text
    if atc:   # registrations such as D-EABC or OE-ABC, spelled out in NATO words
        t = re.sub(r"\b([A-Z]{1,2})-([A-Z]{2,4})\b",
                   lambda m: " ".join(NATO[ch.lower()] for ch in m.group(1) + m.group(2)), t)
    t = t.lower()
    t = NOISE_TAGS.sub(" ", t)
    t = t.replace("??", " ")                                  # unintelligible marker, not scored
    if atc:
        t = t.replace("x-ray", "xray")
        t = re.sub(r"(?<!\d)(\d{1,2}),(?=\d{3}\b)", r"\1", t)   # 1,013 -> 1013 (thousands)
        t = re.sub(r"(?<=\d)[.,](?=\d)", " decimal ", t)      # 121.5 / 118,725 -> … decimal …
    t = re.sub(r"[^\w\s']|_", " ", t)                         # punctuation and hyphens
    t = t.replace("'", "")
    words = []
    for w in t.split():
        if atc:
            w = SPOKEN.get(w, w)
            if any(ch in "0123456789" for ch in w):            # 1013, 4ab, 27l
                for ch in w:
                    words.append(DIGITS[int(ch)] if ch in "0123456789" else NATO.get(ch, ch))
                continue
        words.append(w)
    return words


# --- Alignment ----------------------------------------------------------------

def align(ref, hyp):
    """Levenshtein alignment. Returns a list of (op, ref_token, hyp_token)
    with op in '=', 'S', 'D', 'I'."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def count_ops(ops):
    c = {"=": 0, "S": 0, "D": 0, "I": 0}
    for op, _, _ in ops:
        c[op] += 1
    return c


def char_errors(ref_words, hyp_words):
    ops = align(list(" ".join(ref_words)), list(" ".join(hyp_words)))
    c = count_ops(ops)
    return c["S"] + c["D"] + c["I"], len(" ".join(ref_words))


# --- Running whisper ------------------------------------------------------------

def resolve_model(cfg):
    """Returns the ggml file for a configuration, downloading, converting and
    quantising it first if needed. "model" can be:
      small, base.en, large-v3-turbo-q5_0  whisper.cpp's own ggml models
      hf:<owner>/<repo>                     a fine-tune in transformers format
      https://…/file.bin                    a ggml file to download
      /path/to/ggml-model.bin               a local ggml file"""
    model = cfg["model"]
    if model.startswith("hf:"):
        path = convert_hf(model[3:])
    elif model.startswith(("http://", "https://")):
        path = MODELS / model.split("?")[0].rsplit("/", 1)[-1]
        if not path.is_file():
            download(model, path)
    elif model.endswith(".bin"):
        path = Path(model).expanduser()
        if not path.is_file():
            raise RuntimeError(f"model file not found: {path}")
    else:
        path = MODELS / f"ggml-{model}.bin"
        if not path.is_file():
            download(MODEL_URL.format(model), path)
    return quantize(path, cfg["quantize"]) if cfg.get("quantize") else path


def quantize(src, qtype):
    target = MODELS / f"{src.stem}-{qtype}.bin"
    if target.is_file():
        return target
    if not QUANTIZE_BIN.is_file():
        raise RuntimeError(f"{QUANTIZE_BIN} not found; run the benchmark in the test set's Docker image")
    print(f"  quantising {src.name} to {qtype}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    run_checked([str(QUANTIZE_BIN), str(src), str(target), qtype])
    return target


# Files the converter (or transformers' from_pretrained) may read
HF_FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "vocab.json",
            "added_tokens.json", "merges.txt", "normalizer.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json"]


def convert_hf(repo):
    """Downloads a transformers-format Whisper fine-tune and converts it with
    whisper.cpp's models/convert-h5-to-ggml.py (f16 output)."""
    target = MODELS / f"ggml-{repo.replace('/', '--')}.bin"
    if target.is_file():
        return target
    if not CONVERTER.is_file() or not (OPENAI_WHISPER / "whisper/assets/mel_filters.npz").is_file():
        raise RuntimeError("hf: models need the converter: build the image with WITH_HF=1")
    with urllib.request.urlopen(f"{HF}/api/models/{repo}") as resp:
        files = {s["rfilename"] for s in json.load(resp).get("siblings", [])}
    weights = next((w for w in ("model.safetensors", "pytorch_model.bin") if w in files), None)
    if weights is None:
        raise RuntimeError(f"{repo} has no model.safetensors or pytorch_model.bin")
    src = MODELS / "hf" / repo.replace("/", "--")
    for name in [f for f in HF_FILES if f in files] + [weights]:
        if not (src / name).is_file():
            download(f"{HF}/{repo}/resolve/main/{name}", src / name)

    # The converter reads vocab.json and added_tokens.json directly, and some
    # fine-tunes ship only tokenizer.json. Whisper's vocabulary depends only on
    # the model family, so take it from the matching OpenAI model.
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if not (src / "vocab.json").is_file():
        base = {51864: "openai/whisper-small.en", 51866: "openai/whisper-large-v3"}.get(
            config.get("vocab_size"), "openai/whisper-small")
        download(f"{HF}/{base}/resolve/main/vocab.json", src / "vocab.json")
    if not (src / "added_tokens.json").is_file():
        (src / "added_tokens.json").write_text("{}", encoding="utf-8")

    print(f"  converting {repo} to ggml", flush=True)
    out = src / "out"
    out.mkdir(exist_ok=True)
    run_checked([sys.executable, str(CONVERTER), str(src), str(OPENAI_WHISPER), str(out)])
    os.replace(out / "ggml-model.bin", target)
    shutil.rmtree(src, ignore_errors=True)   # the transformers copy is no longer needed
    return target


def run_checked(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "").strip().splitlines()
        raise RuntimeError(f"{Path(cmd[1] if 'python' in Path(cmd[0]).name else cmd[0]).name} failed: "
                           f"{detail[-1] if detail else res.returncode}")


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}", flush=True)
    tmp = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got >> 20} / {total >> 20} MB", end="", flush=True)
        print()
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def prompt_of(cfg):
    prompt = cfg.get("prompt", True)
    if prompt is True or prompt == "en":
        return DEFAULT_PROMPT
    if prompt == "de":
        return GERMAN_PROMPT
    return prompt or ""


def whisper_cmd(cfg, model_path, wav, opts):
    cmd = [opts.whisper_bin, "-m", str(model_path), "-f", str(wav),
           "-l", cfg.get("lang", "en"), "-t", str(cfg.get("threads", opts.threads)), "-nt", "-np"]
    if prompt_of(cfg):
        cmd += ["--prompt", prompt_of(cfg)]
    cmd += [str(a) for a in cfg.get("args", [])]
    # Low CPU priority, so a benchmark on the recorder's laptop doesn't hold up live transcription
    if opts.nice and shutil.which("nice"):
        cmd = ["nice", "-n", str(opts.nice)] + cmd
    return cmd


def config_key(cfg):
    """Changes whenever anything that affects the output changes."""
    relevant = {k: cfg.get(k) for k in ("model", "lang", "args", "quantize")}
    relevant["prompt"] = prompt_of(cfg)
    relevant["raw"] = bool(cfg.get("raw"))
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]


def run_config(cfg, clips, data, opts, progress=None):
    """Returns {clip_id: {"text", "seconds"}} for one configuration, using the cache.
    progress(done, total), if given, is called after each newly transcribed clip
    (used by the prompt lab in transcribe.py)."""
    cache_path = data / "results" / "runs" / f"{safe_name(cfg['name'])}.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    key = config_key(cfg)
    if cache.get("key") != key or opts.force:
        cache = {"key": key, "config": cfg, "clips": {}}

    todo = [c for c in clips if cache["clips"].get(c["id"], {}).get("source") != c["source"]]
    if todo:
        model_path = resolve_model(cfg)
        for n, clip in enumerate(todo, 1):
            wav = data / clip["audio"]
            started = time.time()
            res = subprocess.run(whisper_cmd(cfg, model_path, wav, opts),
                                 capture_output=True, text=True, timeout=900)
            seconds = time.time() - started
            if res.returncode != 0:
                detail = (res.stderr or "").strip().splitlines()
                raise RuntimeError(f"whisper failed on {wav.name}: {detail[-1] if detail else res.returncode}")
            raw = " ".join(res.stdout.split())
            text = raw if cfg.get("raw") else clean_text(raw, prompt_of(cfg))
            cache["clips"][clip["id"]] = {"source": clip["source"], "text": text,
                                          "raw": raw, "seconds": round(seconds, 2)}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            if progress:
                progress(n, len(todo))
            else:
                print(f"\r  {n}/{len(todo)} clips", end="", flush=True)
        if not progress:
            print()
    return cache["clips"]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# --- Scoring ----------------------------------------------------------------------

def score(name, outputs, clips, atc):
    totals = {"=": 0, "S": 0, "D": 0, "I": 0}
    char_err = char_len = ref_words = 0
    silent = silent_halluc = 0
    seconds = audio = 0.0
    per_clip = {}
    by_lang = {}
    for clip in clips:
        out = outputs.get(clip["id"])
        if out is None:
            continue
        ref = normalize(clip["reference"], atc)
        hyp = normalize(out["text"] or "", atc)
        ops = align(ref, hyp)
        c = count_ops(ops)
        for k in totals:
            totals[k] += c[k]
        ce, cl = char_errors(ref, hyp)
        char_err += ce
        char_len += cl
        ref_words += len(ref)
        if not ref:
            silent += 1
            silent_halluc += bool(hyp)
        if out.get("seconds") is not None:
            seconds += out["seconds"]
            audio += clip.get("duration") or 0
        errs = c["S"] + c["D"] + c["I"]
        lang = clip_language(clip)
        if ref and lang != "none":
            stats = by_lang.setdefault(lang, {"errors": 0, "words": 0})
            stats["errors"] += errs
            stats["words"] += len(ref)
        per_clip[clip["id"]] = {"ops": ops, "errors": errs, "words": len(ref), "text": out["text"],
                                "wer": errs / len(ref) if ref else (1.0 if hyp else 0.0)}
    errors = totals["S"] + totals["D"] + totals["I"]
    return {
        "name": name,
        "clips": len(per_clip),
        "wer": errors / ref_words if ref_words else 0.0,
        "cer": char_err / char_len if char_len else 0.0,
        "sub": totals["S"], "del": totals["D"], "ins": totals["I"], "ref_words": ref_words,
        "exact": sum(1 for p in per_clip.values() if p["errors"] == 0),
        "silent": silent, "silent_halluc": silent_halluc,
        "sec_per_clip": seconds / len(per_clip) if per_clip and audio else None,
        "rtf": seconds / audio if audio else None,
        "per_clip": per_clip,
        "by_lang": by_lang,
    }


def lang_wer(result, lang):
    stats = result["by_lang"].get(lang)
    return stats["errors"] / stats["words"] if stats and stats["words"] else None


def languages(results):
    """Languages with scored words; per-language columns appear only for two or more."""
    found = sorted({l for r in results for l, s in r["by_lang"].items() if s["words"]})
    return found if len(found) > 1 else []


def best_per_language(results, langs):
    """What picking the best configuration for each language would score overall:
    an estimate for detecting the language first and then choosing the model."""
    if not langs:
        return None
    picks, errors, words = {}, 0, 0
    for lang in langs:
        scored = [r for r in results if lang_wer(r, lang) is not None]
        best = min(scored, key=lambda r: lang_wer(r, lang))
        picks[lang] = (best["name"], lang_wer(best, lang))
        errors += best["by_lang"][lang]["errors"]
        words += best["by_lang"][lang]["words"]
    return {"picks": picks, "wer": errors / words if words else 0.0}


def fmt_pct(v):
    return "-" if v is None else f"{v:.1%}"


def describe_combo(combo, results):
    picks = ", ".join(f"{lang}: {name} ({wer:.1%})" for lang, (name, wer) in combo["picks"].items())
    return (f"Best per language: {picks}. Choosing the model by language would give {combo['wer']:.1%} WER "
            f"overall, against {results[0]['wer']:.1%} for the best single configuration "
            f"(excluding the effect of wrong language detection).")


# --- Output ------------------------------------------------------------------------

def print_table(results, langs):
    rows = [("Configuration", "WER") + tuple(f"WER {l}" for l in langs) +
            ("CER", "Sub", "Del", "Ins", "Exact", "Noise→text", "s/clip", "RTF")]
    for r in results:
        rows.append((r["name"], f"{r['wer']:.1%}") + tuple(fmt_pct(lang_wer(r, l)) for l in langs) +
                    (f"{r['cer']:.1%}", r["sub"], r["del"], r["ins"],
                     f"{r['exact']}/{r['clips']}", f"{r['silent_halluc']}/{r['silent']}",
                     "-" if r["sec_per_clip"] is None else f"{r['sec_per_clip']:.1f}",
                     "-" if r["rtf"] is None else f"{r['rtf']:.2f}"))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for n, row in enumerate(rows):
        print("  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w) for i, (v, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  ".join("-" * w for w in widths))
    print("\nWER/CER: word/character error rate (lower is better). Exact: clips without any error.\n"
          "Noise→text: clips you marked as no speech where Whisper still wrote something.\n"
          "RTF: processing time / audio length (below 1 keeps up with live traffic).")
    combo = best_per_language(results, langs)
    if combo:
        print("\n" + describe_combo(combo, results))


def write_csv(results, path, langs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "wer"] + [f"wer_{l}" for l in langs] +
                   ["cer", "substitutions", "deletions", "insertions", "ref_words",
                    "exact_clips", "clips", "noise_clips", "noise_clips_with_text", "sec_per_clip", "rtf"])
        for r in results:
            w.writerow([r["name"], round(r["wer"], 4)] +
                       ["" if lang_wer(r, l) is None else round(lang_wer(r, l), 4) for l in langs] +
                       [round(r["cer"], 4), r["sub"], r["del"], r["ins"],
                        r["ref_words"], r["exact"], r["clips"], r["silent"], r["silent_halluc"],
                        "" if r["sec_per_clip"] is None else round(r["sec_per_clip"], 2),
                        "" if r["rtf"] is None else round(r["rtf"], 3)])


def render_ops(ops):
    out = []
    for op, r, h in ops:
        if op == "=":
            out.append(html.escape(r))
        elif op == "S":
            out.append(f'<span class="s" title="heard: {html.escape(r)}">{html.escape(h)}</span>')
        elif op == "D":
            out.append(f'<span class="d" title="missing">{html.escape(r)}</span>')
        else:
            out.append(f'<span class="i" title="extra">{html.escape(h)}</span>')
    return " ".join(out) or '<span class="muted">(nothing)</span>'


def fmt_rtf(r):
    return "-" if r["rtf"] is None else f"{r['rtf']:.2f}"


def write_report(results, clips, path, configs, langs):
    cfg_by_name = {c["name"]: c for c in configs}
    lang_heads = "".join(f"<th>WER {l}</th>" for l in langs)
    combo = best_per_language(results, langs)
    combo_html = f"<p>{html.escape(describe_combo(combo, results))}</p>" if combo else ""
    summary = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['wer']:.1%}</td>"
        + "".join(f"<td>{fmt_pct(lang_wer(r, l))}</td>" for l in langs) +
        f"<td>{r['cer']:.1%}</td>"
        f"<td>{r['sub']}/{r['del']}/{r['ins']}</td><td>{r['exact']}/{r['clips']}</td>"
        f"<td>{r['silent_halluc']}/{r['silent']}</td>"
        f"<td>{fmt_rtf(r)}</td>"
        f"<td class='muted'>{html.escape(describe(cfg_by_name.get(r['name'])))}</td></tr>"
        for r in results)
    blocks = []
    for clip in clips:
        rows = "".join(
            f"<tr><td>{html.escape(r['name'])}</td>"
            f"<td class='num'>{r['per_clip'][clip['id']]['errors']}</td>"
            f"<td>{render_ops(r['per_clip'][clip['id']]['ops'])}</td></tr>"
            for r in results if clip["id"] in r["per_clip"])
        ref = clip["reference"] or '<span class="muted">(no speech)</span>'
        blocks.append(
            f"<section><h3>#{clip['id']} <span class='muted'>{html.escape(clip['source'])} · "
            f"{(clip.get('duration') or 0):.1f} s · {html.escape(clip_language(clip))}</span></h3>"
            f"<audio controls preload='none' src='../{html.escape(clip['audio'])}'></audio>"
            f"<p><b>You:</b> {html.escape(ref) if clip['reference'] else ref}"
            f"{' <span class=muted>(' + html.escape(clip['notes']) + ')</span>' if clip.get('notes') else ''}</p>"
            f"<table><tr><th>Configuration</th><th>Err</th><th>Output (normalised)</th></tr>{rows}</table></section>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Whisper benchmark</title>
<style>
:root {{ --bg:#f6f7f9; --text:#1d232b; --muted:#68707c; --line:#dde1e6; --s:#b45309; --d:#b91c1c; --i:#1d4ed8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --text:#e4e7eb; --muted:#9aa3ae; --line:#2f353f; --s:#fbbf24; --d:#f87171; --i:#60a5fa; }} }}
body {{ margin:0 auto; max-width:1100px; padding:16px; font:14px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--text); }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 20px; }}
td, th {{ border-bottom:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }}
td:nth-child(3) {{ font-family:ui-monospace, Consolas, monospace; }}
.num {{ text-align:right; }} .muted {{ color:var(--muted); font-weight:normal; }}
.s {{ color:var(--s); text-decoration:underline wavy; }} .d {{ color:var(--d); text-decoration:line-through; }}
.i {{ color:var(--i); font-style:italic; }}
section {{ border-top:2px solid var(--line); }} audio {{ height:32px; }}
</style></head><body>
<h1>Whisper benchmark</h1>
<p class="muted">{time.strftime('%Y-%m-%d %H:%M')} ·{len(clips)} clips · lower WER/CER is better ·
<span class="s">substituted</span> (hover for what you heard), <span class="d">missing</span>, <span class="i">extra</span></p>
<table><tr><th>Configuration</th><th>WER</th>{lang_heads}<th>CER</th><th>Sub/Del/Ins</th><th>Exact</th><th>Noise→text</th><th>RTF</th><th>Settings</th></tr>{summary}</table>
{combo_html}
<h2>Per clip</h2>{''.join(blocks)}
</body></html>"""
    path.write_text(page, encoding="utf-8")


def describe(cfg):
    if not cfg:
        return "transcript stored by the live timeline"
    prompt = cfg.get("prompt", True)
    prompt_label = {True: "ATC prompt", "en": "ATC prompt", "de": "German ATC prompt"}.get(
        prompt, "custom prompt" if prompt else "no prompt")
    parts = [f"model {cfg['model']}", f"lang {cfg.get('lang', 'en')}", prompt_label]
    if cfg.get("quantize"):
        parts.append(f"quantised {cfg['quantize']}")
    if cfg.get("args"):
        parts.append(" ".join(map(str, cfg["args"])))
    if cfg.get("raw"):
        parts.append("no clean-up")
    return ", ".join(parts)


# --- Main ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data"), help="test set folder (from transcribe.py)")
    p.add_argument("--configs", default=str(HERE / "configs.json"))
    p.add_argument("--only", help="comma-separated configuration names to run")
    p.add_argument("--whisper-bin", default=DEFAULT_BIN)
    p.add_argument("--threads", default=os.environ.get("WHISPER_THREADS") or str(os.cpu_count() or 4))
    p.add_argument("--nice", type=int, default=10,
                   help="CPU priority for whisper, 0-19; higher is gentler on the live recorder (default 10, 0 = off)")
    p.add_argument("--force", action="store_true", help="ignore cached results")
    p.add_argument("--no-atc-normalize", action="store_true",
                   help="score on lower-cased words only, without digit/NATO normalisation")
    p.add_argument("--list", action="store_true", help="list configurations and exit")
    opts = p.parse_args()

    data = Path(opts.data)
    configs = json.loads(Path(opts.configs).read_text(encoding="utf-8"))
    if opts.list:
        for c in configs:
            print(f"{c['name']:<28} {describe(c)}")
        return
    if opts.only:
        wanted = {n.strip() for n in opts.only.split(",")}
        configs = [c for c in configs if c["name"] in wanted]
        if not configs:
            sys.exit(f"No configuration named {opts.only}; see --list")

    try:
        testset = json.loads((data / "testset.json").read_text(encoding="utf-8"))
    except OSError:
        sys.exit(f"No test set in {data}. Run transcribe.py first.")
    clips = [c for c in testset["clips"] if c.get("reference") is not None]
    if not clips:
        sys.exit("No clips transcribed yet. Transcribe some with transcribe.py first.")
    counts = {}
    for c in clips:
        counts[clip_language(c)] = counts.get(clip_language(c), 0) + 1
    tagged = ", ".join(f"{n} {lang}" for lang, n in sorted(counts.items()))
    print(f"{len(clips)} transcribed clips of {len(testset['clips'])} ({tagged})\n")
    if not Path(opts.whisper_bin).is_file():
        sys.exit(f"whisper-cli not found at {opts.whisper_bin}; run with: docker compose run --rm benchmark")

    atc = not opts.no_atc_normalize
    results = []
    live = {c["id"]: {"text": c["production_transcript"]} for c in clips
            if c.get("production_transcript") is not None}
    if live:
        results.append(score("live timeline (stored)", live, clips, atc))
    for cfg in configs:
        print(f"{cfg['name']}  ({describe(cfg)})", flush=True)
        try:
            outputs = run_config(cfg, clips, data, opts)
        except Exception as exc:
            print(f"\n  skipped: {exc}")
            continue
        results.append(score(cfg["name"], outputs, clips, atc))

    results.sort(key=lambda r: (r["wer"], r["cer"]))
    langs = languages(results)
    print()
    print_table(results, langs)
    out = data / "results"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(results, out / "summary.csv", langs)
    write_report(results, clips, out / "report.html", configs, langs)
    print(f"\nWrote {out / 'summary.csv'} and {out / 'report.html'}"
          f" (also at /report while transcribe.py runs)")


if __name__ == "__main__":
    main()
