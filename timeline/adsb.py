"""
ADS-B snapshots for the timeline: which aircraft were near the airport when a
transmission was recorded.

Asks several free ADS-B networks at once (and a local readsb, if configured),
merges their answers per aircraft (one network may know the callsign, another
the registration), and keeps the aircraft within ADSB_RADIUS_KM of the airport
and below ADSB_MAX_ALT_FT. Also turns callsigns into the way they are spoken
("Delta Echo Kilo Lima Mike", "Lufthansa four alpha bravo") for Whisper's
prompt.

Uses only the Python standard library. Settings come from environment variables;
without AIRPORT_LAT and AIRPORT_LON it is switched off.
"""
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --- Settings -------------------------------------------------------------------
def _float(name):
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return None


AIRPORT_LAT = _float("AIRPORT_LAT")
AIRPORT_LON = _float("AIRPORT_LON")
ENABLED = AIRPORT_LAT is not None and AIRPORT_LON is not None
RADIUS_KM = float(os.environ.get("ADSB_RADIUS_KM", "20"))
MAX_ALT_FT = float(os.environ.get("ADSB_MAX_ALT_FT", "5000"))
# Aircraft on the ground count only this close to the airport, so a busy airport
# nearby (Düsseldorf from Mönchengladbach) doesn't fill every snapshot
GROUND_KM = float(os.environ.get("ADSB_GROUND_KM", "3"))
SOURCES = [s.strip() for s in os.environ.get("ADSB_SOURCES", "adsb.lol,adsb.fi,opensky").split(",") if s.strip()]
# A local readsb/tar1090 (e.g. http://adsb.local/tar1090/data/aircraft.json), asked first
LOCAL_URL = os.environ.get("ADSB_LOCAL_URL", "").strip()
TIMEOUT = float(os.environ.get("ADSB_TIMEOUT", "6"))
AIRLINES_FILE = Path(os.environ.get("AIRLINES_FILE", "/config/airlines.txt"))
USER_AGENT = "airband-recorder timeline (personal, non-commercial)"
# ------------------------------------------------------------------------------

READSB_URLS = {
    "adsb.lol": "https://api.adsb.lol/v2/point/{lat}/{lon}/{nm}",
    "adsb.fi": "https://opendata.adsb.fi/api/v3/lat/{lat}/lon/{lon}/dist/{nm}",
}
OPENSKY_URL = ("https://opensky-network.org/api/states/all"
               "?lamin={lamin}&lomin={lomin}&lamax={lamax}&lomax={lomax}")
FT_PER_M = 3.28084


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def distance_km(lat, lon):
    """Great-circle distance from the airport."""
    p1, p2 = math.radians(AIRPORT_LAT), math.radians(lat)
    dp, dl = p2 - p1, math.radians(lon - AIRPORT_LON)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# --- Sources -------------------------------------------------------------------------
# Each returns a list of aircraft as {hex, flight, reg, type, alt, ground, lat, lon,
# gs, track}. Answers are reused for a short while, so a burst of transmissions
# doesn't hit the networks (adsb.fi allows one request per second) more often
# than needed; OpenSky's anonymous quota of 400 requests a day is spread evenly.

class Source:
    min_interval = 3.0

    def __init__(self, name):
        self.name = name
        self.lock = threading.Lock()
        self.last = None            # (time, aircraft)
        self.blocked_until = 0.0

    def get(self):
        with self.lock:
            now = time.time()
            if self.last and now - self.last[0] < self.min_interval:
                return self.last[1], "reused"
            if now < self.blocked_until:
                return (self.last[1] if self.last else []), "waiting for quota"
            aircraft = self.fetch()
            self.last = (time.time(), aircraft)
            return aircraft, "ok"

    def request(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.load(resp), resp.headers


class ReadsbSource(Source):
    """adsb.lol, adsb.fi and a local readsb all answer in readsb's JSON format."""

    def __init__(self, name, url):
        super().__init__(name)
        self.url = url

    def fetch(self):
        nm = math.ceil(RADIUS_KM / 1.852) + 1
        data, _ = self.request(self.url.format(lat=AIRPORT_LAT, lon=AIRPORT_LON, nm=nm))
        result = []
        for a in data.get("ac") or data.get("aircraft") or []:
            if a.get("lat") is None or a.get("lon") is None:
                continue
            alt = a.get("alt_baro")
            result.append({
                "hex": (a.get("hex") or "").lower().lstrip("~"),
                "flight": (a.get("flight") or "").strip(),
                "reg": (a.get("r") or "").strip(),
                "type": (a.get("t") or "").strip(),
                "ground": alt == "ground",
                "alt": 0 if alt == "ground" else (round(alt) if isinstance(alt, (int, float)) else None),
                "lat": a["lat"], "lon": a["lon"],
                "gs": a.get("gs"), "track": a.get("track"),
            })
        return result


class OpenSkySource(Source):
    """OpenSky Network, anonymous: callsigns but no registrations, 400 requests a day."""
    min_interval = 60.0

    def fetch(self):
        dlat = RADIUS_KM / 111.0
        dlon = RADIUS_KM / (111.0 * math.cos(math.radians(AIRPORT_LAT)))
        try:
            data, headers = self.request(OPENSKY_URL.format(
                lamin=round(AIRPORT_LAT - dlat, 3), lomin=round(AIRPORT_LON - dlon, 3),
                lamax=round(AIRPORT_LAT + dlat, 3), lomax=round(AIRPORT_LON + dlon, 3)))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:     # quota used up: wait as long as OpenSky says
                retry = exc.headers.get("X-Rate-Limit-Retry-After-Seconds")
                self.blocked_until = time.time() + (float(retry) if retry else 3600)
            raise
        # Spread the remaining daily quota over the rest of the UTC day
        remaining = headers.get("X-Rate-Limit-Remaining")
        if remaining is not None and remaining.isdigit():
            left_today = 86400 - time.time() % 86400
            self.min_interval = max(60.0, left_today / max(int(remaining) - 5, 1))
        result = []
        for s in data.get("states") or []:
            if s[5] is None or s[6] is None:
                continue
            alt_m = s[7] if s[7] is not None else s[13]
            result.append({
                "hex": (s[0] or "").lower(), "flight": (s[1] or "").strip(), "reg": "", "type": "",
                "ground": bool(s[8]), "alt": 0 if s[8] else (round(alt_m * FT_PER_M) if alt_m is not None else None),
                "lat": s[6], "lon": s[5],
                "gs": round(s[9] * 1.94384, 1) if s[9] is not None else None, "track": s[10],
            })
        return result


_sources = None


def sources():
    global _sources
    if _sources is None:
        _sources = []
        if LOCAL_URL:
            _sources.append(ReadsbSource("local", LOCAL_URL))
        for name in SOURCES:
            if name in READSB_URLS:
                _sources.append(ReadsbSource(name, READSB_URLS[name]))
            elif name == "opensky":
                _sources.append(OpenSkySource(name))
            else:
                log(f"ADS-B: unknown source {name!r} ignored (known: {', '.join(list(READSB_URLS) + ['opensky'])})")
    return _sources


def snapshot():
    """Asks all sources at once and returns {at, sources, aircraft}: the merged
    aircraft near the airport, nearest first."""
    srcs = sources()
    status, lists = {}, []
    with ThreadPoolExecutor(max_workers=max(len(srcs), 1)) as pool:
        futures = {src.name: pool.submit(src.get) for src in srcs}
        for name, fut in futures.items():
            try:
                aircraft, how = fut.result(timeout=TIMEOUT + 2)
                status[name] = f"{how}, {len(aircraft)} aircraft"
                lists.append((name, aircraft))
            except Exception as exc:     # one network failing must not stop the others
                status[name] = f"error: {str(exc)[:120]}"
    return {"at": int(time.time()), "sources": status, "aircraft": merge(lists)}


def merge(lists):
    """One entry per aircraft (ICAO hex); each field from the first source that has it."""
    merged = {}
    for name, aircraft in lists:
        for a in aircraft:
            key = a["hex"] or f"{name}:{a['flight']}"
            m = merged.setdefault(key, {"hex": a["hex"], "sources": []})
            m["sources"].append(name)
            for field in ("flight", "reg", "type", "alt", "ground", "lat", "lon", "gs", "track"):
                if m.get(field) in (None, "") and a.get(field) not in (None, ""):
                    m[field] = a[field]
    result = []
    for m in merged.values():
        if m.get("lat") is None:
            continue
        dist = distance_km(m["lat"], m["lon"])
        if dist > RADIUS_KM:
            continue
        if m.get("ground") and dist > GROUND_KM:
            continue
        if m.get("alt") is not None and m["alt"] > MAX_ALT_FT:
            continue
        result.append({
            "hex": m["hex"], "flight": m.get("flight", ""), "reg": m.get("reg", ""), "type": m.get("type", ""),
            "alt": m.get("alt"), "ground": bool(m.get("ground")), "dist": round(dist, 1),
            "lat": round(m["lat"], 4), "lon": round(m["lon"], 4),
            "gs": m.get("gs"), "track": m.get("track"), "sources": m["sources"],
        })
    result.sort(key=lambda a: a["dist"])
    return result


# --- How callsigns are spoken ----------------------------------------------------------

NATO = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India",
        "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
        "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray", "Yankee", "Zulu"]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# ICAO airline designator -> radio telephony designator. Add more in
# config/airlines.txt, one "DLH = Lufthansa" per line.
BUILTIN_AIRLINES = {
    "DLH": "Lufthansa", "GEC": "Lufthansa Cargo", "CLH": "Hansaline", "EWG": "Eurowings",
    "CFG": "Condor", "TUI": "TUI jet", "SXS": "Sunexpress", "BEL": "Beeline", "AUA": "Austrian",
    "SWR": "Swiss", "KLM": "KLM", "AFR": "Airfrance", "BAW": "Speedbird", "EZY": "Easy",
    "EJU": "Alpine", "RYR": "Ryanair", "WZZ": "Wizz Air", "VLG": "Vueling", "IBE": "Iberia",
    "SAS": "Scandinavian", "FIN": "Finnair", "LOT": "Lot", "THY": "Turkish", "PGT": "Sunturk",
    "UAE": "Emirates", "QTR": "Qatari", "BCS": "Eurotrans", "DHK": "World Express",
    "UPS": "UPS", "FDX": "Fedex", "GAF": "German Air Force", "CHX": "Christoph",
    "NJE": "Fraction", "EXS": "Channex", "TOM": "Tomson", "BTI": "Air Baltic",
}
_airlines = {"mtime": None, "map": None}


def airlines():
    """{ICAO designator: telephony}, built-in and from AIRLINES_FILE (re-read when it changes)."""
    try:
        mtime = AIRLINES_FILE.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _airlines["map"] is not None and _airlines["mtime"] == mtime:
        return _airlines["map"]
    table = dict(BUILTIN_AIRLINES)
    if mtime is not None:
        for n, line in enumerate(AIRLINES_FILE.read_text(encoding="utf-8").splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            code, sep, name = (part.strip() for part in line.partition("="))
            if not sep or not re.fullmatch(r"[A-Za-z]{3}", code) or not name:
                log(f"{AIRLINES_FILE.name} line {n} ignored, expected 'DLH = Lufthansa': {line}")
                continue
            table[code.upper()] = name
    _airlines.update(mtime=mtime, map=table)
    return table


def spell(chars, letters_capitalised=True):
    words = []
    for ch in chars.upper():
        if ch.isdigit():
            words.append(DIGITS[int(ch)])
        elif "A" <= ch <= "Z":
            word = NATO[ord(ch) - 65]
            words.append(word if letters_capitalised else word.lower())
    return " ".join(words)


def spoken(aircraft):
    """How the aircraft's callsign is said on the radio, or "" if unknown:
    "Delta Echo Kilo Lima Mike" for D-EKLM, "Lufthansa four alpha bravo" for DLH4AB."""
    flight, reg = aircraft.get("flight", ""), aircraft.get("reg", "")
    reg_plain = reg.replace("-", "").upper()
    m = re.fullmatch(r"([A-Z]{3})(\d[0-9A-Z]{0,3})", flight.upper())
    if m and flight.upper() != reg_plain:
        name = airlines().get(m.group(1))
        return f"{name} {spell(m.group(2), False)}" if name else ""
    if reg_plain:                     # GA: the registration, letter by letter
        return spell(reg_plain)
    if re.fullmatch(r"[A-Z]{1,2}[A-Z0-9]{3,4}", flight.upper()):
        return spell(flight)          # a registration sent as the flight ID
    return ""


def prompt_for(aircraft, base, limit=8):
    """Whisper's prompt for one clip: the usual prompt, then the callsigns of the
    nearest aircraft (last, where they matter most for Whisper)."""
    names = []
    for a in aircraft or []:
        s = spoken(a)
        if s and s not in names:
            names.append(s)
        if len(names) >= limit:
            break
    if not names:
        return base
    return f"{base} {', '.join(names)}.".strip()
