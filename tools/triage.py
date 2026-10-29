#!/usr/bin/env python3
"""triage — local web app to keep or drop the models found by hf_roots.py.

Serves tools/triage.html on http://127.0.0.1:8765 (localhost only) and reads
the last hf_roots run (~/.cache/hf_roots/last_run.json, written by every run).
Each decision is written at once to data/triage.json:

  {"<family>": {"decision": "keep" | "drop", "reason": "...", "id": ..., "date": ...}}

hf_roots.py applies it on its next run: "drop" sets the family aside, "keep"
selects a root for README (section "Selected in triage") or keeps a
derivative as an exception. The "Régénérer ROOTS.md" button re-runs hf_roots
from its cache.

Descriptions are the first paragraphs of each model card, fetched on demand
from huggingface.co and cached in ~/.cache/hf_roots/cards/. Benchmark scores
are extracted from the card's tables (markdown or HTML): the column (or row)
whose header best matches the model name gives the values, benchmark names are
normalised (GPQA-Diamond, "GPQA Diamond (AA)" -> GPQA Diamond). Automatic
extraction: every value carries the header and row it came from, to check it
against the card. "Scanner les fiches" fetches all cards in the background so
that the Benchmarks tab can compare every model.

Usage:
  tools/triage.py              # opens the browser
  tools/triage.py --port 9000 --no-browser
"""

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Data

def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_triage(path, decisions):
    """One decision per line, sorted, written atomically."""
    lines = [f"  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False, sort_keys=True)}"
             for k, v in sorted(decisions.items())]
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w") as f:
        f.write("{\n" + ",\n".join(lines) + ("\n" if lines else "") + "}\n")
    os.replace(tmp, path)


def from_config(value):
    """ignored/kept strings set by config.toml, as opposed to by triage."""
    return bool(value) and not str(value).startswith("triage")


def models(args):
    run = load_json(os.path.join(args.cache_dir, "last_run.json"), {"families": {}})
    out = []
    for f in run["families"].values():
        if not f.get("first_reported"):
            continue  # under the report threshold, like ROOTS.md
        if from_config(f.get("ignored")) or from_config(f.get("kept")):
            continue  # already decided in config.toml
        if f["kind"] == "derivative" and f["listed"]:
            pile = "readme"
        elif f["listed"]:
            pile = "listed"  # only shown in the Benchmarks tab
        else:
            pile = "candidates"
        out.append({k: f.get(k) for k in (
            "id", "family", "publisher", "domain", "modalities", "pipeline", "params", "active",
            "q4_gb", "tier", "license", "license_class", "likes", "created", "latest", "arch",
            "kind", "why", "suspect_base", "related_listed", "variants", "gated")}
            | {"pile": pile})
    return {"generated": run.get("generated"), "models": out,
            "decisions": load_json(args.triage, {})}


# --------------------------------------------------------------------------
# Model card description

def fetch_card(repo, cache_dir):
    """Raw model card, cached; None + error when unavailable."""
    path = os.path.join(cache_dir, "cards", hashlib.sha256(repo.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        c = load_json(path, {})
        if "raw" in c or c.get("error"):
            return c
    url = f"https://huggingface.co/{repo}/raw/main/README.md"
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "hf_roots-triage/1"}), timeout=30) as r:
            c = {"raw": r.read().decode("utf-8", "replace"), "source": url}
    except urllib.error.HTTPError as e:
        c = {"raw": None, "source": url,
             "error": "carte privée (modèle gated)" if e.code in (401, 403) else f"HTTP {e.code}"}
    except OSError as e:
        return {"raw": None, "source": url, "error": str(e)}  # not cached: retry later
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(c, f, ensure_ascii=False)
    return c


def card_cached(repo, cache_dir):
    path = os.path.join(cache_dir, "cards", hashlib.sha256(repo.encode()).hexdigest() + ".json")
    return load_json(path, None) if os.path.exists(path) else None


def card_text(repo, cache_dir):
    c = fetch_card(repo, cache_dir)
    raw = c.get("raw")
    return {"text": summarize(raw) if raw else None, "source": c.get("source"),
            "error": c.get("error"), "benches": benchmarks(raw, repo) if raw else []}


def summarize(md, limit=900):
    """First prose paragraphs of a model card: no front matter, code, tables,
    headings, badges, link bars, footnote lists or HTML."""
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"<(table|style|script).*?</\1>", "", md, flags=re.S | re.I)
    md = re.sub(r"<br\s*/?>", "\n", md, flags=re.I)
    md = re.sub(r"<[^>]+>", " ", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)            # images
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)        # links -> text
    md = html.unescape(md).replace("\u00a0", " ")
    out = []
    for p in re.split(r"\n\s*\n", md):
        lines = [l.strip() for l in p.splitlines()]
        lines = [l for l in lines if l and not l.startswith(("#", "|", ">", "-", "*", "+", "[", "<", "!"))
                 and l.count(" | ") < 2 and l.count("|") < 3 and l.count(" · ") < 3]
        p = re.sub(r"\*\*|__|`", "", " ".join(lines)).strip()
        if len(p) < 60 or sum(c.isalpha() for c in p) < 0.6 * len(p):
            continue
        out.append(p)
        if sum(map(len, out)) > limit / 2 or len(out) == 3:
            break
    text = " ".join(out)
    return (text[:limit].rsplit(" ", 1)[0] + " …") if len(text) > limit else (text or None)


# --------------------------------------------------------------------------
# Benchmarks from model card tables

# canonical name, pattern on the lowercased cell; first match wins
BENCH_NAMES = [
    ("SuperGPQA", r"supergpqa"), ("GPQA Diamond", r"gpqa"),
    ("MMLU-ProX", r"mmlu.?prox"), ("MMLU-Pro", r"mmlu.?pro"), ("MMLU-Redux", r"mmlu.?redux"),
    ("MMLU", r"^mmlu\b"),
    ("AIME 2026", r"aime.{0,3}(?:20)?26"), ("AIME 2025", r"aime.{0,3}(?:20)?25"),
    ("AIME 2024", r"aime.{0,3}(?:20)?24"), ("HMMT", r"hmmt"), ("IMO-AnswerBench", r"imo.?answer"),
    ("MATH-500", r"math.?500"), ("GSM8K", r"gsm.?8k"),
    ("LiveCodeBench v6", r"(?:livecodebench|lcb).*v6"), ("LiveCodeBench", r"livecodebench|^lcb"),
    ("SWE-bench Verified", r"swe.?(?:bench.?)?verified"), ("SWE-bench Pro", r"swe.?bench.?pro"),
    ("SWE-bench Multilingual", r"swe.?bench.?multi"), ("DeepSWE", r"deepswe"),
    ("Terminal-Bench 2.1", r"terminal.?bench.{0,3}2\.1"), ("Terminal-Bench 2.0", r"terminal.?bench.{0,3}2(?:\.0)?\b"),
    ("Terminal-Bench", r"terminal.?bench"),
    ("HLE", r"^hle\b|humanity.?s last exam"), ("IFEval", r"^ifeval"), ("IFBench", r"ifbench"),
    ("BFCL v4", r"bfcl.{0,3}v?4"), ("BFCL v3", r"bfcl.{0,3}v?3"), ("BFCL", r"bfcl"),
    ("τ²-Bench", r"tau.?2|τ²|τ2"), ("MMMU-Pro", r"mmmu.?pro"), ("MMMU", r"^mmmu"),
    ("MathVista", r"mathvista"), ("OCRBench", r"ocrbench"), ("DocVQA", r"docvqa"),
    ("ChartQA", r"chartqa"), ("Video-MME", r"video.?mme"), ("OmniDocBench", r"omnidocbench"),
    ("RULER", r"ruler"), ("HumanEval", r"humaneval"), ("MBPP", r"mbpp"), ("Codeforces", r"codeforces"),
    ("OSWorld", r"osworld"), ("BrowseComp", r"browsecomp"), ("ScreenSpot", r"screenspot"),
    ("Arena-Hard", r"arena.?hard"), ("MultiChallenge", r"multichallenge"), ("SimpleQA", r"simpleqa"),
    ("LongBench v2", r"longbench.{0,3}v?2"), ("AA-LCR", r"aa.?lcr"),
    ("AA Intelligence Index", r"intelligence index|^aa index"),
]
GENERIC = {"instruct", "it", "bf16", "fp8", "fp16", "base", "hf", "chat", "preview",
           "thinking", "model", "the", "a", "nvidia", "gguf"}


# not percentages: any value allowed; every other score must be within 0-100
# (cards also tabulate question counts: "GPQA-Diamond | 448")
UNBOUNDED = {"Codeforces"}


def plausible(bench, v):
    return v is not None and (bench in UNBOUNDED or 0 <= v <= 100)


def bench_name(cell):
    low = cell.lower().strip()
    for name, pat in BENCH_NAMES:
        if re.search(pat, low):
            return name
    return None


def clean_cell(c):
    c = re.sub(r"<br\s*/?>", " ", c, flags=re.I)
    c = re.sub(r"<[^>]+>", "", c)
    c = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", c)
    c = re.sub(r"\*\*|__|`|\\", "", c)
    return " ".join(c.replace("&nbsp;", " ").split())


def tables(md):
    """[(header, rows)] from markdown pipe tables and HTML tables."""
    out = []
    lines = []
    for l in md.splitlines():
        # a pipe row cut by a newline inside a cell ("Muse Glimmer-30B\nHigh Reasoning")
        if lines and lines[-1].lstrip().startswith("|") and not lines[-1].rstrip().endswith("|") \
                and not l.lstrip().startswith("|"):
            lines[-1] += " " + l.strip()
        else:
            lines.append(l)
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append([clean_cell(c) for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append((block[0], block[2:]))
            continue
        i += 1
    for t in re.findall(r"<table.*?</table>", md, flags=re.S | re.I):
        rows = []
        for tr in re.findall(r"<tr.*?</tr>", t, flags=re.S | re.I):
            row = []
            for attrs, c in re.findall(r"<t[hd]([^>]*)>(.*?)</t[hd]>", tr, flags=re.S | re.I):
                span = re.search(r"colspan\s*=\s*[\"']?(\d+)", attrs)
                row += [clean_cell(c)] * (int(span.group(1)) if span else 1)
            rows.append(row)
        rows = [r for r in rows if r]
        if len(rows) > 1:
            out.append((rows[0], rows[1:]))
    return out


def name_tokens(s):
    s = s.lower()
    s = re.sub(r"(?<=[a-z]{2})(?=\d)", " ", s)        # open2 -> open 2, qwen3 -> qwen 3
    s = re.sub(r"(\d+[bm])(?=[a-z]{2,})", r"\1 ", s)  # 30bhigh -> 30b high
    return {t for t in re.split(r"[^a-z0-9]+", s) if t and t not in GENERIC}


def header_score(header, model):
    """How well a header cell names this model; digits must not contradict
    (Qwen3.6-27B is not Qwen3.8-27B). An active-size token (a15b) only counts
    against when the model name has one."""
    if re.search(r"^ours\b|\(ours\)|this model", header.strip(), flags=re.I):
        return 1.0
    h = name_tokens(header)
    if not h:
        return 0
    has_active = any(re.fullmatch(r"a\d+(?:\.\d+)?b", t) for t in model)
    wrong = {t for t in h - model if re.search(r"\d", t)
             and (has_active or not re.fullmatch(r"a\d+(?:\.\d+)?b", t))}
    return len(h & model) / len(model) - 0.5 * len(wrong)


def number(cell):
    if re.fullmatch(r"\s*[-–—]*\s*", cell):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", cell.replace(",", ""))
    return float(m.group(0)) if m else None


def cell_at(row, header, j):
    """Cell of `row` under header column j; rows shortened by a rowspan
    (category column) are aligned from the right."""
    if len(row) == len(header):
        return row[j]
    k = len(row) - (len(header) - j)
    return row[k] if 0 <= k < len(row) else ""


def row_label(row):
    for c in row[:2]:
        b = bench_name(c) if c else None
        if b:
            return b, c
    return None, None


def benchmarks(md, repo):
    """[{bench, value, raw, label, header}] for this model, one per benchmark
    (first occurrence wins). A table counts only if it yields >= 2 known
    benchmarks for the matched column (or row)."""
    model = name_tokens(repo.split("/", 1)[1])
    found = {}
    for head0, rows0 in tables(md):
        allrows = [head0] + rows0
        best = None  # (score, orientation, header index, column/row index)
        width = max(map(len, allrows))
        for k in range(min(4, len(allrows) - 1)):
            header = allrows[k]
            # a header row shortened by rowspans may name a model in its 1st cell
            for j in range(0 if len(header) < width else 1, len(header)):
                sc = header_score(header[j], model)
                if best is None or sc > best[0]:
                    best = (sc, "col", k, j)
        if (len(head0) == 2 and (best is None or best[0] <= 0)
                and not re.search(r"descr|detail|note|task|domain", head0[1], flags=re.I)):
            best = (0.5, "col", 0, 1)  # "Benchmark | Score" table of a single model
        for i, r in enumerate(rows0):
            if r:
                sc = header_score(r[0], model)
                if best is None or sc > best[0]:
                    best = (sc, "row", 0, i)
        if not best or best[0] < 0.5:
            continue
        got = {}
        if best[1] == "col":
            header = allrows[best[2]]
            for r in allrows[best[2] + 1:]:
                b, label = row_label(r)
                raw = cell_at(r, header, best[3]) if b else ""
                v = number(raw) if raw else None
                if b and plausible(b, v) and b not in got:
                    got[b] = {"bench": b, "value": v, "raw": raw, "label": label, "header": header[best[3]]}
        else:
            r = rows0[best[3]]
            for j, h in enumerate(head0[1:], 1):
                b = bench_name(h)
                raw = cell_at(r, head0, j) if b else ""
                v = number(raw) if raw else None
                if b and plausible(b, v) and b not in got:
                    got[b] = {"bench": b, "value": v, "raw": raw, "label": h, "header": r[0]}
        if len(got) >= 2:
            for b, x in got.items():
                found.setdefault(b, x)
    return list(found.values())


SCAN = {"running": False, "done": 0, "total": 0}


def scan_cards(args):
    """Fetch every missing card in the background (raw files, polite pace)."""
    ids = [m["id"] for m in models(args)["models"]]
    todo = [i for i in ids if card_cached(i, args.cache_dir) is None]
    SCAN.update(running=True, done=0, total=len(todo))

    def one(i):
        fetch_card(i, args.cache_dir)
        with LOCK:
            SCAN["done"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, todo))
    SCAN["running"] = False


def all_benches(args):
    out = {}
    for m in models(args)["models"]:
        c = card_cached(m["id"], args.cache_dir)
        if c and c.get("raw"):
            out[m["family"]] = benchmarks(c["raw"], m["id"])
    return {"benches": out, "scan": SCAN}


# --------------------------------------------------------------------------
# HTTP

class Handler(BaseHTTPRequestHandler):
    args = None

    def log_message(self, fmt, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/":
            with open(os.path.join(HERE, "triage.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/models":
            return self.send(200, models(self.args))
        if path == "/api/card":
            repo = urllib.parse.unquote(query.removeprefix("id="))
            if not REPO_RE.match(repo):
                return self.send(400, {"error": "bad id"})
            return self.send(200, card_text(repo, self.args.cache_dir))
        if path == "/api/benches":
            return self.send(200, all_benches(self.args))
        self.send(404, {"error": "not found"})

    def do_POST(self):
        # the page is same-origin; refuse anything posted from another site
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.args.port}",
                                              f"http://localhost:{self.args.port}"):
            return self.send(403, {"error": "origin"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.send(400, {"error": "bad json"})
        if self.path == "/api/decide":
            fam, decision = req.get("family"), req.get("decision")
            if not isinstance(fam, str) or decision not in ("keep", "drop", None):
                return self.send(400, {"error": "bad request"})
            with LOCK:
                d = load_json(self.args.triage, {})
                if decision is None:
                    d.pop(fam, None)
                else:
                    d[fam] = {"decision": decision, "id": req.get("id"),
                              "reason": (req.get("reason") or "").strip() or None,
                              "date": time.strftime("%Y-%m-%d")}
                write_triage(self.args.triage, d)
            return self.send(200, {"ok": True, "decisions": d})
        if self.path == "/api/scan-cards":
            if not SCAN["running"]:
                threading.Thread(target=scan_cards, args=(self.args,), daemon=True).start()
            return self.send(200, {"ok": True})
        if self.path == "/api/apply":
            # re-render from the cache only: decisions do not need fresh data
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "hf_roots.py"),
                                "--ttl-hours", "1000000", "--cache-dir", self.args.cache_dir],
                               capture_output=True, text=True, timeout=600)
            return self.send(200 if p.returncode == 0 else 500,
                             {"ok": p.returncode == 0, "log": (p.stdout + p.stderr)[-4000:]})
        self.send(404, {"error": "not found"})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--triage", default=os.path.join(ROOT, "data", "triage.json"))
    p.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/hf_roots"))
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if not os.path.exists(os.path.join(args.cache_dir, "last_run.json")):
        sys.exit("no hf_roots run found: run tools/hf_roots.py first")
    Handler.args = args
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"triage: {url}  (Ctrl-C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
