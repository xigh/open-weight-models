#!/usr/bin/env python3
"""hf_roots — track the root models published on Hugging Face by followed labs.

A *root* model is one published by the lab that trained it, not a fine-tune,
quantization, adapter or merge of someone else's model. Hugging Face encodes
derivation as `base_model:{finetune,quantized,adapter,merge}:<repo>` tags:

  root            no base_model tag at all
  variant         base_model in the same lab and the same family (Instruct of
                  its own Base, FP8 of its own BF16...) — folded into the family
  lab-derivative  base_model in the same lab but a new family (a model the lab
                  derived from its own: pruned, distilled, RL-tuned) — listed
                  in its own section, apart from the roots
  derivative      base_model from another author — dropped

Undeclared derivatives are caught by three extra signals:
  - the author is a known repackager (unsloth, bartowski, mlx-community...)
  - the repo name carries a quantization/merge marker (GGUF, AWQ, abliterated...)
  - the exact safetensors parameter count matches an older model from another
    author (a fine-tune keeps its base's shape) — reported as "suspect"

config.toml (edited by hand) lists the followed organisations, the README
criteria and the families set aside. Every model of the followed organisations
released since the date floor is scanned; a global scan sorted by likes is used
to suggest organisations worth following.

Every run is merged into data/models.json (one entry per family per line,
first_seen / last_seen / first_reported tracked), and ROOTS.md is regenerated
from it: families that reached the report threshold since the previous run
first, then the candidates per tier, newest first, then what README.md already
lists.

Stdlib only (Python >= 3.11 for tomllib). Raw API responses are cached
(--cache-dir, --ttl-hours) so a run is reproducible and cheap to re-render.
Unauthenticated API quota is 500 requests / 5 min; HF_TOKEN (or
~/.cache/huggingface/token) is used if present.

Usage:
  tools/hf_roots.py            # updates data/models.json + ROOTS.md
  tools/hf_roots.py -v         # log every organisation scanned
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"
EXPAND = ["createdAt", "tags", "safetensors", "likes", "downloads",
          "pipeline_tag", "cardData", "gated"]

# Pipelines covered by the list (LLMs, VLMs, OCR, any-to-any). None = untagged.
LLM_PIPELINES = {"text-generation", "image-text-to-text", "any-to-any",
                 "text2text-generation", "image-to-text", "video-text-to-text",
                 "audio-text-to-text", None}

# Accounts that republish other labs' weights (quantized, converted, mirrored).
REPACKAGERS = {
    "unsloth", "bartowski", "lmstudio-community", "mlx-community", "thebloke",
    "quantfactory", "mradermacher", "maziyarpanahi", "second-state", "gaianet",
    "ggml-org", "comfy-org", "kijai", "lightx2v", "city96", "redhatai",
    "neuralmagic", "nm-testing", "ubergarm", "devquasar", "noctrex",
    "huihui-ai", "cognitivecomputations", "turboderp", "lonestriker",
    "solidrust", "casperhansen", "intel", "amd", "openvino", "onnx-community",
    "xenova", "llmware", "janhq", "nexaai", "tensorblock", "matrixportal",
    "gguf-org", "calcuis", "localai", "ollama", "mistral-community",
}

# Name markers of a derived repo (quantization, merge, uncensoring, LoRA...).
DERIVED_RE = re.compile(
    r"(?i)(^|[-_.])(gguf|ggml|awq|gptq|exl2|exl3|mlx|onnx|openvino|"
    r"\d+bit|\d+bpw|q[2-8]_k|iq[1-4]|abliterated|uncensored|heretic|"
    r"merge[d]?|lora|qlora)([-_.]|$)")

# Q4 footprint ~0.55 bytes/param (DeepSeek-V4-Flash 284B -> ~156 GB, README).
Q4_BYTES_PER_PARAM = 0.55

# license (cardData.license, or license_name when license is "other") -> class.
LICENSE_CLASS = {
    "apache-2.0": "open", "mit": "open", "bsd-3-clause": "open",
    "bsd-2-clause": "open", "bsd": "open", "cc-by-4.0": "open",
    "cc-by-sa-4.0": "open", "openmdw-1.1": "open", "openmdw": "open",
    "nvidia-open-model-license": "open", "nvidia-oml": "open",
    "llama3": "llama", "llama3.1": "llama", "llama3.2": "llama",
    "llama3.3": "llama", "llama4": "eu-excluded",
    "tencent-hunyuan-community": "eu-excluded",
    "openrail": "conditional", "openrail++": "conditional",
    "creativeml-openrail-m": "conditional", "bigscience-openrail-m": "conditional",
    "gemma": "conditional", "deepseek": "conditional", "qwen": "conditional",
    "jamba-open-model-license": "conditional", "modified-mit": "conditional",
    "lfm1.0": "conditional", "lfm-open-license-v1.0": "conditional",
    "falcon-llm-license": "conditional",
    "cc-by-nc-4.0": "non-commercial", "cc-by-nc-sa-4.0": "non-commercial",
    "cc-by-nc-nd-4.0": "non-commercial", "qwen-research": "non-commercial",
    "mistral-research": "non-commercial", "mnpl": "non-commercial",
}

FAMILY_SUFFIX_RE = re.compile(
    r"(?i)[-_.](instruct|it|base|chat|thinking|reasoning|hf|bf16|fp16|fp8|fp4|"
    r"nvfp4|mxfp4|int4|int8|w4a16|w8a8|preview|exp|experimental|pytorch|"
    r"diffusers|transformers|original|dynamic|\d+gb|\d{4}|\d{6}|\d{8}|"
    # speculative-decoding drafters ship with (and are folded into) their model
    r"assistant|drafter|dspark|dflash|eagle\d*|mtp)$")

SIZE_RE = re.compile(r"(?i)(?:^|[-_.])(?:(\d+)x)?(\d+(?:\.\d+)?)([mbt])(?=$|[-_.])")
ACTIVE_RE = re.compile(r"(?i)(?:^|[-_.])a(\d+(?:\.\d+)?)b(?=$|[-_.])")

VERBOSE = False


def log(msg, always=False):
    if VERBOSE or always:
        print(msg, file=sys.stderr, flush=True)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# HTTP with on-disk cache and rate-limit pacing

class Client:
    def __init__(self, cache_dir, ttl_hours):
        self.cache_dir = cache_dir
        self.ttl = ttl_hours * 3600
        self.requests = 0
        self.token = os.environ.get("HF_TOKEN")
        tok = os.path.expanduser("~/.cache/huggingface/token")
        if not self.token and os.path.exists(tok):
            self.token = open(tok).read().strip() or None
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, url):
        """Return (json_body, next_url). Cached responses keep their Link."""
        path = os.path.join(self.cache_dir,
                            hashlib.sha256(url.encode()).hexdigest() + ".json")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < self.ttl:
            with open(path) as f:
                c = json.load(f)
            return c["body"], c["next"]
        for attempt in range(6):
            req = urllib.request.Request(url, headers={"User-Agent": "hf_roots/1"})
            if self.token:
                req.add_header("Authorization", "Bearer " + self.token)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = json.load(r)
                    headers = r.headers
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return [], None
                if e.code == 429 or e.code >= 500:
                    wait = _ratelimit_reset(e.headers) or 30 * (attempt + 1)
                    log(f"HTTP {e.code}, waiting {wait}s", always=True)
                    time.sleep(wait)
                    continue
                raise
        else:
            raise RuntimeError(f"giving up on {url}")
        self.requests += 1
        nxt = _next_link(headers.get("Link", ""))
        with open(path, "w") as f:
            json.dump({"url": url, "fetched": now_iso(), "next": nxt,
                       "body": body}, f)
        # Pace ourselves before the quota runs out instead of eating 429s.
        rl = headers.get("RateLimit", "")
        m = re.search(r"r=(\d+);t=(\d+)", rl)
        if m and int(m.group(1)) < 5:
            log(f"quota low ({rl}), sleeping {m.group(2)}s", always=True)
            time.sleep(int(m.group(2)) + 1)
        return body, nxt

    def paginate(self, params, stop=None, max_pages=1000):
        q = urllib.parse.urlencode(params + [("expand[]", e) for e in EXPAND])
        url = f"{API}?{q}"
        for _ in range(max_pages):
            body, url = self.get(url)
            for m in body:
                if stop and stop(m):
                    return
                yield m
            if not url:
                return


def _next_link(link):
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return m.group(1) if m else None


def _ratelimit_reset(headers):
    m = re.search(r"t=(\d+)", headers.get("RateLimit", "") if headers else "")
    return int(m.group(1)) + 1 if m else None


# --------------------------------------------------------------------------
# Classification

def author_of(repo):
    return repo.split("/", 1)[0]


def base_models(m):
    """[(relation, repo)] from base_model:<rel>:<repo> tags (+ cardData)."""
    out = []
    for t in m.get("tags") or []:
        if t.startswith("base_model:"):
            parts = t.split(":", 2)
            if len(parts) == 3:
                out.append((parts[1], parts[2]))
            elif len(parts) == 2 and not any(r == parts[1] for _, r in out):
                out.append(("unknown", parts[1]))
    bm = (m.get("cardData") or {}).get("base_model")
    for r in ([bm] if isinstance(bm, str) else bm or []):
        if isinstance(r, str) and not any(x == r for _, x in out):
            out.append(("unknown", r))
    return out


def same_lab(a, b, aliases):
    a, b = a.lower(), b.lower()
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    # two accounts mapped to the same display name are the same lab
    return aliases.get(a) is not None and aliases.get(a) == aliases.get(b)


def classify(m, aliases):
    """(kind, reason) with kind in root / variant / lab-derivative / derivative."""
    repo = m["id"]
    author = author_of(repo)
    if author.lower() in REPACKAGERS:
        return "derivative", "repackager account"
    if DERIVED_RE.search(repo.split("/", 1)[1]):
        return "derivative", "derived-name marker"
    bases = [(rel, r) for rel, r in base_models(m) if r != repo]
    if not bases:
        return "root", ""
    foreign = [(rel, r) for rel, r in bases if not same_lab(author, author_of(r), aliases)]
    if foreign:
        rel, r = foreign[0]
        return "derivative", f"{rel} of {r}"
    fam = family_of(repo)
    if any(family_of(r) == fam for _, r in bases):
        return "variant", "of " + bases[0][1]
    return "lab-derivative", f"{bases[0][0]} of {bases[0][1]}"


def params_of(m):
    """(total_params, active_params, source). Safetensors first, name second."""
    name = m["id"].split("/", 1)[1]
    active = None
    a = ACTIVE_RE.search(name)
    if a:
        active = float(a.group(1)) * 1e9
    st = m.get("safetensors") or {}
    if st.get("total"):
        return st["total"], active, "safetensors"
    best = None
    for mult, num, unit in SIZE_RE.findall(name):
        v = float(num) * {"m": 1e6, "b": 1e9, "t": 1e12}[unit.lower()]
        v *= int(mult) if mult else 1
        if active and abs(v - active) < 1:
            continue
        best = max(best or 0, v)
    return best, active, "name" if best else None


def license_of(m):
    cd = m.get("cardData") or {}
    lic = cd.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if not lic:
        lic = next((t[8:] for t in m.get("tags") or [] if t.startswith("license:")), None)
    name = cd.get("license_name")
    key = (name if lic == "other" and name else lic) or ""
    cls = LICENSE_CLASS.get(key.lower(), "unknown" if not key else "check")
    return key or "?", cls


def family_of(repo):
    author, name = repo.split("/", 1)
    name = name.lower()
    while True:
        new = FAMILY_SUFFIX_RE.sub("", name)
        if new == name:
            break
        name = new
    return f"{author.lower()}/{name}"


def tier_of(params, crit):
    if not params:
        return "unknown"
    gb = params / 1e9 * Q4_BYTES_PER_PARAM
    if gb <= crit["main_q4_gb"]:
        return "main"
    if gb <= crit["extended_q4_gb"]:
        return "extended"
    return "beyond"


# --------------------------------------------------------------------------
# README comparison

def readme_repos(path):
    if not path or not os.path.exists(path):
        return set()
    text = open(path).read()
    return {f"{o}/{r}" for o, r in re.findall(r"huggingface\.co/([\w.-]+)/([\w.-]+)", text)
            if o not in ("datasets", "spaces", "papers", "docs", "blog")}


def tokens(family):
    return re.split(r"[-_.]", family.split("/", 1)[1])


def listed_match(family, readme_families):
    """Exact family match, else a README family of the same author sharing the
    two leading name tokens (mistral-small-4-128k ~ mistral-small-4-119b)."""
    if family in readme_families:
        return family, True
    author = family.split("/", 1)[0]
    tk = tokens(family)
    for rf in sorted(readme_families):
        if rf.split("/", 1)[0] != author:
            continue
        rt = tokens(rf)
        n = 0
        while n < min(len(tk), len(rt)) and tk[n] == rt[n]:
            n += 1
        if n >= 3 or (n >= 2 and n == min(len(tk), len(rt))):
            return rf, False
    return None, False


# --------------------------------------------------------------------------
# Collection and analysis

def collect(client, cfg):
    crit, disc = cfg["criteria"], cfg["discovery"]
    pool = {}
    log(f"global scan by likes (>= {disc['min_likes']})", always=True)
    for m in client.paginate([("sort", "likes"), ("direction", "-1"), ("limit", "1000")],
                             stop=lambda m: m.get("likes", 0) < disc["min_likes"],
                             max_pages=200):
        pool.setdefault(m["id"], m)
    log(f"  {len(pool)} models", always=True)
    orgs = sorted(cfg["orgs"])
    log(f"followed organisations: {len(orgs)}", always=True)
    for i, a in enumerate(orgs):
        n0 = len(pool)
        # newest first, stop at the date floor: older releases cannot qualify
        for m in client.paginate([("author", a), ("sort", "createdAt"),
                                  ("direction", "-1"), ("limit", "1000")],
                                 stop=lambda m: m.get("createdAt", "") < crit["since"]):
            pool.setdefault(m["id"], m)
        log(f"  [{i + 1}/{len(orgs)}] {a}: +{len(pool) - n0}")
    log(f"  {len(pool)} models, {client.requests} HTTP requests (rest from cache)",
        always=True)
    return pool


def analyse(pool, cfg, listed_repos):
    crit = cfg["criteria"]
    followed = {a.lower(): name for a, name in cfg["orgs"].items()}
    ignore = {k.lower(): v for k, v in cfg.get("ignore", {}).items()}

    # root models per exact parameter count, oldest first, for the suspect check
    by_params = {}
    for m in sorted(pool.values(), key=lambda m: m.get("createdAt", "")):
        total = (m.get("safetensors") or {}).get("total")
        if total and total > 1e8 and classify(m, followed)[0] == "root":
            by_params.setdefault(total, []).append(m)

    readme_families = {family_of(r) for r in listed_repos}
    stats = {"pool": len(pool), "root": 0, "variant": 0, "lab-derivative": 0,
             "derivative": 0, "followed_models": 0}
    rows, unfollowed = [], {}
    for m in pool.values():
        if m.get("createdAt", "") < crit["since"]:
            continue
        if crit["pipelines"] != "all" and m.get("pipeline_tag") not in LLM_PIPELINES:
            continue
        total, active, psrc = params_of(m)
        # untagged repos are mostly non-LLM artefacts; keep only sized ones
        if (crit["pipelines"] != "all" and m.get("pipeline_tag") is None
                and (total or 0) < 3e8):
            continue
        kind, why = classify(m, followed)
        author = author_of(m["id"])
        suspect = None
        if kind == "root" and total in by_params:
            older = [o for o in by_params[total]
                     if o["id"] != m["id"] and o.get("createdAt", "") < m.get("createdAt", "")]
            # a lab re-using the shape of its own older model is not suspect
            if not any(same_lab(author_of(o["id"]), author, followed) for o in older):
                suspect = older[0]["id"] if older else None
        if author.lower() not in followed:
            if kind == "root" and not suspect:
                unfollowed.setdefault(author, []).append(m)
            continue
        stats["followed_models"] += 1
        stats[kind] += 1
        if kind == "derivative":
            continue
        lic, lcls = license_of(m)
        fam = family_of(m["id"])
        match, exact = listed_match(fam, readme_families)
        rows.append({
            "id": m["id"], "author": author, "publisher": followed[author.lower()],
            "family": fam, "kind": kind, "why": why, "suspect_base": suspect,
            "created": m.get("createdAt", "")[:10], "pipeline": m.get("pipeline_tag"),
            "likes": m.get("likes", 0), "downloads": m.get("downloads", 0),
            "params": total, "active": active, "params_source": psrc,
            "q4_gb": round(total / 1e9 * Q4_BYTES_PER_PARAM, 1) if total else None,
            "tier": tier_of(total, crit), "license": lic, "license_class": lcls,
            "gated": bool(m.get("gated")),
            "listed": m["id"] in listed_repos or exact,
            "related_listed": None if exact else match,
            "ignored": ignore.get(m["id"].lower()) or ignore.get(fam),
        })

    suggestions = []
    for author, ms in unfollowed.items():
        top = max(ms, key=lambda m: m.get("likes", 0))
        if top.get("likes", 0) >= cfg["discovery"]["suggest_min_likes"]:
            suggestions.append({"author": author, "roots": len(ms),
                                "likes": sum(m.get("likes", 0) for m in ms),
                                "top": top["id"], "top_likes": top.get("likes", 0),
                                "latest": max(m.get("createdAt", "") for m in ms)[:10]})
    suggestions.sort(key=lambda s: -s["likes"])
    return rows, stats, suggestions


KIND_RANK = {"root": 0, "lab-derivative": 1, "variant": 2}


def group_families(rows):
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    out = []
    for fam, members in fams.items():
        # representative: a root if there is one, then the most liked repo
        members.sort(key=lambda r: (KIND_RANK[r["kind"]], -r["likes"], r["id"]))
        rep = dict(members[0])
        if rep["kind"] == "variant":
            # the family's root is outside the scan window: it is a lab derivative
            rep["kind"] = "lab-derivative"
        rep["variants"] = sorted(r["id"] for r in members[1:])
        rep["likes"] = sum(r["likes"] for r in members)
        rep["downloads"] = sum(r["downloads"] for r in members)
        rep["created"] = min(r["created"] for r in members)
        rep["latest"] = max(r["created"] for r in members)
        rep["listed"] = any(r["listed"] for r in members)
        rep["ignored"] = next((r["ignored"] for r in members if r["ignored"]), None)
        out.append(rep)
    return out


def merge_db(path, fams, stats, suggestions, generated, min_likes):
    """Fold this run into the persistent database; return (db, newly reported).

    A family is *reported* once its likes reach the report threshold; the run
    where that first happens is its `first_reported`, which drives the "New
    since previous run" section (a release with 3 likes today shows up as new
    on the run where it becomes visible)."""
    old_db = {}
    if os.path.exists(path):
        with open(path) as f:
            old_db = json.load(f)
    families = old_db.get("families", {})
    new = set()
    for r in fams:
        prev = families.get(r["family"]) or {}
        entry = dict(r)
        entry["first_seen"] = prev.get("first_seen") or generated
        entry["last_seen"] = generated
        entry["first_reported"] = prev.get("first_reported")
        if not entry["first_reported"] and r["likes"] >= min_likes:
            entry["first_reported"] = generated
            new.add(r["family"])
        families[r["family"]] = entry
    db = {"generated": generated, "previous_run": old_db.get("generated"),
          "stats": stats, "suggested_orgs": suggestions,
          "families": dict(sorted(families.items()))}
    return db, new


def write_db(path, db):
    """JSON with one family / suggestion per line, so git diffs stay readable."""
    def dump(v):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    out = ["{"]
    for k in ("generated", "previous_run", "stats"):
        out.append(f" {dump(k)}: {dump(db[k])},")
    out.append(' "suggested_orgs": [')
    out.append(",\n".join("  " + dump(s) for s in db["suggested_orgs"]))
    out.append(" ],")
    out.append(' "families": {')
    out.append(",\n".join(f"  {dump(k)}: {dump(v)}" for k, v in db["families"].items()))
    out.append(" }")
    out.append("}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


# --------------------------------------------------------------------------
# Rendering

def fmt_params(p):
    if not p:
        return "?"
    if p >= 1e12:
        return f"{p / 1e12:.2f}T"
    if p >= 1e9:
        return f"{p / 1e9:.1f}B" if p < 1e11 else f"{p / 1e9:.0f}B"
    return f"{p / 1e6:.0f}M"


LICENSE_MARK = {"open": "", "llama": " (EU: text-only)", "conditional": " ⚠️",
                "check": " ❓", "unknown": " ❓", "non-commercial": " ⛔",
                "eu-excluded": " ⛔ EU"}


def md_table(fams):
    lines = ["| Model | Publisher | Released | Total | Active | Q4 | Pipeline | License | Likes | Note |",
             "|-------|-----------|----------|-------|--------|----|----------|---------|------:|------|"]
    for r in fams:
        note = []
        if r["kind"] == "lab-derivative":
            note.append(r["why"])
        if r["suspect_base"]:
            note.append(f"⚠️ same param count as `{r['suspect_base']}`")
        if r["related_listed"]:
            note.append(f"related to listed `{r['related_listed']}`")
        if r["variants"]:
            note.append(f"+{len(r['variants'])} variant(s)")
        if r["gated"]:
            note.append("gated")
        if r["params_source"] == "name":
            note.append("size from name")
        if r["ignored"]:
            note.append(f"set aside: {r['ignored']}")
        q4 = f"{r['q4_gb']:.0f} GB" if r["q4_gb"] else "?"
        released = r["created"] if r["created"] == r["latest"] else f"{r['created']} → {r['latest']}"
        lines.append(
            f"| [{r['id']}](https://huggingface.co/{r['id']}) | {r['publisher']} | {released} "
            f"| {fmt_params(r['params'])} | {fmt_params(r['active']) if r['active'] else '--'} "
            f"| {q4} | {r['pipeline'] or '--'} "
            f"| {r['license']}{LICENSE_MARK[r['license_class']]} | {r['likes']} "
            f"| {'; '.join(note)} |")
    return "\n".join(lines)


def sort_recent(fams):
    return sorted(fams, key=lambda f: (f["latest"], f["likes"]), reverse=True)


def render_md(db, cfg, new):
    crit, disc = cfg["criteria"], cfg["discovery"]
    # families not seen by this run (deleted, renamed) stay in the database only;
    # below the report threshold they are tracked but not shown
    fams = [f for f in db["families"].values()
            if f["last_seen"] == db["generated"] and f["first_reported"]]
    live = [f for f in fams if not f["ignored"]]
    roots = [f for f in live if f["kind"] == "root" and not f["suspect_base"]]
    unlisted = [f for f in roots if not f["listed"]]

    def pick(tiers, lcls):
        return sort_recent([f for f in unlisted if f["tier"] in tiers and f["license_class"] in lcls])

    ok, cond = {"open", "llama"}, {"conditional", "check", "unknown"}
    bad = {"non-commercial", "eu-excluded"}
    s = db["stats"]
    new_rows = sort_recent([f for f in live if f["family"] in new])
    out = [
        "# Root models on Hugging Face",
        "",
        "Generated by [`tools/hf_roots.py`](tools/hf_roots.py) from [`data/models.json`](data/models.json); "
        "organisations, criteria and set-aside models are in [`config.toml`](config.toml). Do not edit this "
        "file: re-run the tool. This is the candidate list for [README.md](README.md) — every entry still needs "
        "a human read of its model card before it goes into the curated list.",
        "",
        f"- Last run: {db['generated']} (previous: {db['previous_run'] or 'none — this run is the baseline'})",
        f"- Scope: every model released since {crit['since']} by the {len(cfg['orgs'])} followed organisations, "
        f"pipelines {'all' if crit['pipelines'] == 'all' else 'LLM / VLM / any-to-any'}.",
        f"- Their models in scope: {s['followed_models']} → {s['root']} roots, {s['variant']} variants "
        f"(folded into their family), {s['lab-derivative']} lab derivatives, {s['derivative']} third-party "
        f"derivatives or repackagings (dropped).",
        "- One row per family (`-Instruct`, `-Base`, `-FP8`, drafters, date suffixes… folded), newest release "
        f"first; likes summed over the family. Shown once the family reaches {crit['report_min_likes']} likes "
        f"({len(db['families'])} families tracked in the database, {len(fams)} shown).",
        f"- Q4 ≈ {Q4_BYTES_PER_PARAM} bytes/param; main tier ≤ {crit['main_q4_gb']} GB, extended ≤ "
        f"{crit['extended_q4_gb']} GB. Total params from safetensors metadata, else parsed from the repo name "
        "(packed FP4/INT4 checkpoints may under-report).",
        "- **Root** = no `base_model` tag. **Lab derivative** = `base_model` is another family of the same lab "
        "(pruned, distilled, RL-tuned from its own model). ⚠️ *same param count* = no declared base but exactly "
        "the parameter count of an older model from another author: probably an undeclared fine-tune.",
        "",
    ]
    out += [f"## New since previous run ({len(new_rows)})", ""]
    if db["previous_run"]:
        out += [md_table(new_rows) if new_rows else "_none_", ""]
    else:
        out += ["_First run: baseline. Every family below is new._", ""]
    sections = [
        ("Main tier (Q4 ≤ {m} GB), open license", pick({"main", "unknown"}, ok)),
        ("Extended tier ({m} < Q4 ≤ {e} GB), open license", pick({"extended"}, ok)),
        ("License to check (conditional / custom / unknown), Q4 ≤ {e} GB",
         pick({"main", "extended", "unknown"}, cond)),
        ("Beyond {e} GB Q4 (reference only)", pick({"beyond"}, ok | cond)),
        ("Non-commercial or EU-excluded (would be rejected)",
         pick({"main", "extended", "beyond", "unknown"}, bad)),
        ("Lab derivatives (new family derived from the lab's own model)",
         sort_recent([f for f in live if f["kind"] == "lab-derivative" and not f["listed"]])),
        ("Suspected undeclared fine-tunes",
         sort_recent([f for f in live if f["suspect_base"] and not f["listed"]])),
        ("Already in README", sort_recent([f for f in live if f["listed"]])),
        ("Set aside in config.toml", sort_recent([f for f in fams if f["ignored"]])),
    ]
    for title, sel in sections:
        title = title.format(m=crit["main_q4_gb"], e=crit["extended_q4_gb"])
        out += [f"## {title} ({len(sel)})", "", md_table(sel) if sel else "_none_", ""]

    sug = db["suggested_orgs"]
    out += [f"## Suggested organisations ({len(sug)})", "",
            f"Not followed, but published a root model with ≥ {disc['suggest_min_likes']} likes "
            f"(global scan down to {disc['min_likes']} likes). Add the worthwhile ones to `[orgs]` in "
            "`config.toml`; individuals re-uploading fine-tunes without declaring a base also land here.", ""]
    if sug:
        out += ["| Organisation | Roots | Likes | Latest | Most liked |",
                "|--------------|------:|------:|--------|------------|"]
        out += [f"| [{s['author']}](https://huggingface.co/{s['author']}) | {s['roots']} | {s['likes']} "
                f"| {s['latest']} | [{s['top']}](https://huggingface.co/{s['top']}) ({s['top_likes']}) |"
                for s in sug]
    else:
        out += ["_none_"]
    out.append("")
    return "\n".join(out)


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    crit = {"since": "2024-04-01", "main_q4_gb": 128, "extended_q4_gb": 256, "pipelines": "llm",
            "report_min_likes": 50}
    crit.update(cfg.get("criteria", {}))
    disc = {"min_likes": 100, "suggest_min_likes": 300}
    disc.update(cfg.get("discovery", {}))
    cfg["criteria"], cfg["discovery"] = crit, disc
    cfg.setdefault("orgs", {})
    cfg.setdefault("ignore", {})
    return cfg


def main():
    global VERBOSE
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(root, "config.toml"))
    p.add_argument("--readme", default=os.path.join(root, "README.md"))
    p.add_argument("--db", default=os.path.join(root, "data", "models.json"))
    p.add_argument("--out-md", default=os.path.join(root, "ROOTS.md"))
    p.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/hf_roots"))
    p.add_argument("--ttl-hours", type=float, default=24,
                   help="reuse cached API responses younger than this (default 24)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    VERBOSE = args.verbose

    cfg = load_config(args.config)
    pool = collect(Client(args.cache_dir, args.ttl_hours), cfg)
    rows, stats, suggestions = analyse(pool, cfg, readme_repos(args.readme))
    fams = group_families(rows)
    db, new = merge_db(args.db, fams, stats, suggestions, now_iso(),
                       cfg["criteria"]["report_min_likes"])
    write_db(args.db, db)
    with open(args.out_md, "w") as f:
        f.write(render_md(db, cfg, new if db["previous_run"] else set()))
    unlisted = sum(1 for r in fams if not r["listed"] and r["kind"] == "root")
    log(f"{len(fams)} families ({unlisted} roots not in README, {len(new)} newly reported) "
        f"-> {args.out_md}, {args.db}", always=True)


if __name__ == "__main__":
    main()
