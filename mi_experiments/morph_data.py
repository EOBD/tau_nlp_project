"""Build the Hebrew verb-morphology datasets for probing: verbs in context (UD) and a form lexicon (Wiktionary).

    python -m mi_experiments.morph_data [--out data/morph] [--refresh]

Downloads to <out>/raw/ (skipped if present): UD_Hebrew-HTB, -IAHLTwiki, -IAHLTknesset (train/dev/test) and the
kaikki.org English-Wiktionary Hebrew extract; Hebrew Wiktionary pages are fetched through its API only for verbs no
other source covers, and cached in <out>/raw/hewiktionary_pages.json. Writes:

  ud_verbs.jsonl   one row per VERB token in context: sentence text, character span of the whitespace word
                   (start/end, includes clitics like ו/ש/כש) and of the verb host inside it (host_start/host_end),
                   lemma, binyan, tense, person, gender, number, root, root_source, root_class, lexicon ambiguity
  lexicon.jsonl    one row per (lemma, cell, unvocalized spelling) from the Wiktionary conjugation tables
  weak_roots_to_check.csv
                   lemmas no source gives a root for (their root is null); fill correct_root (letters, any separator)
                   and rerun: filled values are kept across rebuilds and win over everything
  stats.json       counts, and the rule's accuracy per root class against Wiktionary's own roots

Conventions: binyan names follow UD (PAAL NIFAL PIEL PUAL HIFIL HUFAL HITPAEL). tense is one of past present future
imperative infinitive (lexicon also: gerund, passive_participle). Present tense has no person (UD's 1,2,3 / 3 on
participles is dropped). gender "m,f" = common form. Roots are letters joined by "־" with final letters normalised
(כ־ת־ב, ש־מ־ר). root_source: manual (the CSV) > wiktionary (English) > derived (the rule below, only in root classes
where it matches English Wiktionary >= 95%) > hewiktionary > null. Text is unvocalized, as the model sees it.
"""

import argparse
import csv
import json
import os
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict

UD = {t: f"https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-{t}/master/he_{t.lower()}-ud-{{}}.conllu"
      for t in ("HTB", "IAHLTwiki", "IAHLTknesset")}
KAIKKI = "https://kaikki.org/dictionary/Hebrew/kaikki.org-dictionary-Hebrew.jsonl"
BINYAN = {"pa": "PAAL", "nif": "NIFAL", "pi": "PIEL", "pu": "PUAL", "hif": "HIFIL", "huf": "HUFAL", "hit": "HITPAEL"}
FINALS = str.maketrans("ךםןףץ", "כמנפצ")
HEB = re.compile(r"[א-ת]+")
TRUSTED_RULE_ACC = 0.95  # root classes where the rule is at least this accurate on Wiktionary's roots are not flagged


def download(out, refresh):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    jobs = [(u.format(s), os.path.join(raw, os.path.basename(u.format(s))))
            for u in UD.values() for s in ("train", "dev", "test")]
    jobs.append((KAIKKI, os.path.join(raw, "kaikki_hebrew.jsonl")))
    for url, path in jobs:
        if refresh or not os.path.exists(path):
            print("downloading", url)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
    return raw


# ============================================================================================ roots

def letters(s):
    """Consonant skeleton: drop niqqud and anything non-Hebrew, normalise final letters."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if "א" <= c <= "ת").translate(FINALS)


def derive_root(lemma, binyan):
    """Root from the 3ms-past lemma by stripping the binyan's affixes. Right for most strong roots; weak roots (hollow,
    geminate, initial נ/י) come out wrong, e.g. הכיר -> כ־ו־ר (is נ־כ־ר)."""
    w = letters(lemma)
    if len(w) < 2:
        return None
    if binyan == "PAAL":
        r = w
    elif binyan == "NIFAL":
        r = w[1:] if w[0] == "נ" else w
    elif binyan in ("PIEL", "PUAL"):
        r = w[0] + w[2:] if len(w) == 4 and w[1] == ("י" if binyan == "PIEL" else "ו") else w
    elif binyan == "HIFIL":
        r = w[1:]
        r = r[:-2] + r[-1] if len(r) == 4 and r[-2] == "י" else r
    elif binyan == "HUFAL":
        r = w[1:]
        r = r[1:] if len(r) == 4 and r[0] == "ו" else r
    elif binyan == "HITPAEL":
        r = w[2:] if w.startswith("הת") else w[1:]
        if len(r) >= 3 and r[0] in "סשצז" and r[1] in "תטד":  # הסתדר, הצטלם, הזדקן
            r = r[0] + r[2:]
    else:
        return None
    if len(r) == 2:  # hollow guess; also where geminates and dropped נ/י land
        r = r[0] + "ו" + r[1]
    return "־".join(r)


def norm_root(s):
    """Wiktionary/manual root -> ש־מ־ר; None if it is not 2-5 letters (a few Wiktionary root templates are malformed)."""
    r = letters(s)
    return "־".join(r) if 2 <= len(r) <= 5 else None


def root_class(root):
    r = root.split("־") if root else []
    if len(r) != 3:
        return "quadriliteral" if len(r) >= 4 else "unknown"
    a, b, c = r
    if a in "יו":
        return "pe_yod"
    if b in "וי":
        return "hollow"
    if b == c:
        return "geminate"
    if c in "הא":
        return "final_he_alef"
    if a == "נ":
        return "pe_nun"
    if set(r) & set("אהחע"):
        return "guttural"
    return "strong"


# ============================================================================================ Wiktionary

def cell(tags, form):
    """Conjugation-table tags -> (tense, person, gender, number), or None for non-cells."""
    t = set(tags)
    if "error-unrecognized-form" in t:
        return ("infinitive", None, None, None) if form.startswith("ל") else None
    if "negative" in t or "table-tags" in t or "inflection-template" in t:
        return None
    if "noun-from-verb" in t:
        tense = "gerund"
    elif "participle" in t and "passive" in t:
        tense = "passive_participle"
    else:
        tense = next((x for x in ("past", "present", "future", "imperative") if x in t), None)
        if tense is None:
            return None
    person = next((p[0] for p in ("1first-person", "2second-person", "3third-person") if p[1:] in t), None)
    gender = ",".join(g[0] for g in ("masculine", "feminine") if g in t) or None
    number = "sg" if "singular" in t else "pl" if "plural" in t else None
    return tense, person if tense not in ("present", "passive_participle") else None, gender, number


def load_wiktionary(path):
    """Verb headwords -> {(lemma letters, binyan): {"root": str|None, "cells": {cell: set of spellings}, ...}}.
    Spellings are unvocalized plene forms: the table's own unvocalized entries and the page titles its vocalized
    entries link to; only if neither exists, the vocalized form with niqqud stripped (defective spelling)."""
    lex = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("pos") != "verb" or not r.get("head_templates") or r["head_templates"][0]["name"] != "he-verb":
            continue
        binyan = BINYAN.get(r["head_templates"][0]["args"].get("1"))
        if binyan is None:
            continue
        root = next((t["args"].get("1") for t in r.get("etymology_templates", [])
                     if t["name"] in ("he-rootbox", "he-root") and t["args"].get("1")), None)
        plene, stripped = defaultdict(set), defaultdict(set)
        for f in r.get("forms", []):
            if f.get("source") != "conjugation":
                continue
            c = cell(f.get("tags", []), f["form"])
            if c is None:
                continue
            if HEB.fullmatch(f["form"]):
                plene[c].add(f["form"])
            for target, page in f.get("links", []):
                page = page.split("#")[0]
                if HEB.fullmatch(page):
                    plene[c].add(page)
            s = "".join(ch for ch in unicodedata.normalize("NFD", f["form"]) if "א" <= ch <= "ת")
            if s:
                stripped[c].add(s)
        cells = {c: (plene[c], "plene") if plene[c] else (stripped[c], "stripped") for c in set(plene) | set(stripped)}
        key = (letters(r["word"]), binyan)
        entry = lex.setdefault(key, {"lemma": r["word"], "binyan": binyan, "root": None, "cells": {},
                                     "gloss": ((r.get("senses") or [{}])[0].get("glosses") or [None])[0]})
        entry["root"] = entry["root"] or (norm_root(root) if root else None)
        for c, v in cells.items():
            entry["cells"].setdefault(c, v)
    return lex


# ============================================================================================ Hebrew Wiktionary

HEWIKT_API = "https://he.wiktionary.org/w/api.php"
HE_BINYAN = {"קל": "PAAL", "פעל": "PAAL", "פָּעַל": "PAAL", "נפעל": "NIFAL", "נִפְעַל": "NIFAL", "פיעל": "PIEL",
             "פִּעֵל": "PIEL", "פועל": "PUAL", "פֻּעַל": "PUAL", "הפעיל": "HIFIL", "הִפְעִיל": "HIFIL",
             "הופעל": "HUFAL", "הפעל": "HUFAL", "הֻפְעַל": "HUFAL", "התפעל": "HITPAEL", "הִתְפַּעֵל": "HITPAEL"}
REDIRECT = re.compile(r"^\s*#(?:הפניה|REDIRECT)\s*\[\[([^\]|#]+)", re.I)


def fetch_hewiktionary(cache_path, titles):
    """Page wikitext by title ("" = no such page), cached in cache_path; only titles not in the cache are fetched.
    Redirect pages are stored as their redirect text and resolved by the caller."""
    import time
    import urllib.error
    import urllib.parse
    pages = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    todo = sorted(set(titles) - set(pages))
    if todo:
        print(f"fetching {len(todo)} Hebrew Wiktionary pages")
    i, wait = 0, 3
    while i < len(todo):
        batch = todo[i:i + 20]
        data = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                       "format": "json", "formatversion": 2, "maxlag": 5,
                                       "titles": "|".join(batch)}).encode()
        req = urllib.request.Request(HEWIKT_API, data=data,
                                     headers={"User-Agent": "tau-nlp-research/0.1 (university student research project)"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:  # 429 / maxlag: back off
            wait = min(max(wait * 2, int(e.headers.get("Retry-After") or 0)), 300)
            time.sleep(wait)
            continue
        norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
        got = {norm.get(p["title"], p["title"]): p.get("revisions", [{}])[0].get("slots", {}).get("main", {})
               .get("content", "") for p in d["query"]["pages"]}
        for t in batch:
            pages[t] = got.get(t, "")
        with open(cache_path + ".part", "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False)
        os.replace(cache_path + ".part", cache_path)
        i, wait = i + 20, 3
        time.sleep(3)
    return pages


def parse_hewiktionary(text):
    """[(binyan, root)] from the verb grammar boxes ({{ניתוח דקדוקי לפועל|...|שורש וגזרה={{שרש3|נ|ג|ע}}|בניין=הפעיל}})."""
    out = []
    for blk in re.split(r"\{\{ניתוח דקדוקי", text)[1:]:
        blk = blk[:1500]
        b = re.search(r"\|\s*בניין\s*=\s*\[*([^\n|\]}]*)", blk)
        r = re.search(r"\{\{שרש(\d)\|([^}]*)\}\}", blk)
        if not (b and r):
            continue
        name = b.group(1).split("#")[-1].replace("(קל)", "").strip()
        binyan = HE_BINYAN.get(name)
        # positional args only, and only as many as the template's letter count (drops homograph numbers, גזרה=...)
        root = norm_root("".join([x.strip() for x in r.group(2).split("|") if "=" not in x][:int(r.group(1))]))
        if binyan and root:
            out.append((binyan, root))
    return out


def spelling_variants(w):
    """w plus its defective spellings (Hebrew Wiktionary titles many verbs without the ו/י vowel letters): drop any
    subset of inner ו/י, collapse יי/וו."""
    import itertools
    out = {w, w.replace("יי", "י").replace("וו", "ו")}
    idx = [i for i, c in enumerate(w) if c in "וי" and 0 < i < len(w) - 1]
    for k in range(1, len(idx) + 1):
        for comb in itertools.combinations(idx, k):
            out.add("".join(c for i, c in enumerate(w) if i not in comb))
    return out


class HeWiktionary:
    """Root lookup by (lemma, binyan). A root is returned only if exactly one root is listed for that binyan: first
    on the page titled with the lemma itself, else across its defective spellings (the binyan must still match, which
    keeps a variant from hitting an unrelated word)."""

    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.pages = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    def text(self, title):
        t = self.pages.get(title, "")
        m = REDIRECT.match(t)
        return self.pages.get(m.group(1).strip(), "") if m else t

    def needed_titles(self, lemmas):
        """Titles to fetch for these lemmas: the lemma itself, then its variants if the lemma has no verb box, then
        redirect targets. Call fetch() until this is empty."""
        out = set()
        for lem in lemmas:
            if lem not in self.pages:
                out.add(lem)
            elif not parse_hewiktionary(self.text(lem)):
                out |= {v for v in spelling_variants(lem) if v not in self.pages}
        for t in list(self.pages):
            m = REDIRECT.match(self.pages[t])
            if m and m.group(1).strip() not in self.pages:
                out.add(m.group(1).strip())
        return out

    def fetch(self, lemmas):
        while True:
            need = self.needed_titles(lemmas)
            if not need:
                return
            self.pages = fetch_hewiktionary(self.cache_path, need)

    def candidates(self, lemma, binyan):
        c = {r for b, r in parse_hewiktionary(self.text(lemma)) if b == binyan}
        if c:
            return c
        return {r for v in spelling_variants(lemma) for b, r in parse_hewiktionary(self.text(v)) if b == binyan}

    def root(self, lemma, binyan):
        c = self.candidates(lemma, binyan)
        return next(iter(c)) if len(c) == 1 else None


# ============================================================================================ UD

def ud_feats(s):
    return dict(x.split("=", 1) for x in s.split("|") if "=" in x)


def ud_tense(f):
    if f.get("VerbForm") == "Inf":
        return "infinitive"
    if f.get("Mood") == "Imp":
        return "imperative"
    if f.get("VerbForm") == "Part" or f.get("Tense") == "Pres":
        return "present"
    return {"Past": "past", "Fut": "future"}.get(f.get("Tense"))


def read_conllu(path):
    """Yield (sent_id, text, rows, surface) per sentence; surface maps word id -> (start, end, surface form)."""
    sid, text, rows = None, None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip()
        elif line.startswith("# text"):
            text = line.split("=", 1)[1].strip()
        elif line and not line.startswith("#"):
            rows.append(line.split("\t"))
        elif not line and rows:
            yield sid, text, rows, surface_spans(text, rows)
            sid, text, rows = None, None, []


def surface_spans(text, rows):
    spans, pos, skip_to = {}, 0, 0
    for c in rows:
        if "." in c[0]:
            continue
        if "-" in c[0]:
            a, b = map(int, c[0].split("-"))
            ids, skip_to = range(a, b + 1), b
        elif int(c[0]) <= skip_to:
            continue
        else:
            ids = [int(c[0])]
        i = text.find(c[1], pos)
        if i < 0:
            continue
        pos = i + len(c[1])
        for k in ids:
            spans[k] = (i, pos, c[1])
    return spans


# ============================================================================================ build

def load_manual(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {(letters(row["lemma"]), row["binyan"]): norm_root(row["correct_root"])
                for row in csv.DictReader(f) if norm_root(row.get("correct_root", ""))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/morph")
    ap.add_argument("--refresh", action="store_true", help="re-download the raw files")
    args = ap.parse_args()
    raw = download(args.out, args.refresh)
    check_path = os.path.join(args.out, "weak_roots_to_check.csv")
    manual = load_manual(check_path)
    lex = load_wiktionary(os.path.join(raw, "kaikki_hebrew.jsonl"))

    # how reliable is the rule, per class of the root it produces? (scored against Wiktionary's roots)
    rule_n, rule_ok = Counter(), Counter()
    for (lem, b), e in lex.items():
        if e["root"]:
            d = derive_root(lem, b)
            rule_n[root_class(d)] += 1
            rule_ok[root_class(d)] += d == e["root"]
    rule_acc = {c: rule_ok[c] / rule_n[c] for c in rule_n}
    trusted = {c for c, a in rule_acc.items() if a >= TRUSTED_RULE_ACC and rule_n[c] >= 20}

    # Hebrew Wiktionary, for the verbs neither a person, English Wiktionary nor a trusted rule class covers
    ud_lemmas = {(c[2], ud_feats(c[5]).get("HebBinyan")) for url in UD.values() for split in ("train", "dev", "test")
                 for _, _, rows, _ in read_conllu(os.path.join(raw, os.path.basename(url.format(split))))
                 for c in rows if c[3] == "VERB" and "-" not in c[0] and "." not in c[0]}
    all_lemmas = ud_lemmas | {(e["lemma"], b) for (_, b), e in lex.items()}
    hewikt = HeWiktionary(os.path.join(raw, "hewiktionary_pages.json"))

    def rule_ok(lem, b):
        return root_class(derive_root(lem, b)) in trusted

    def uncovered(lem, b):
        k = (letters(lem), b)
        return k not in manual and not (k in lex and lex[k]["root"]) and not rule_ok(lem, b)

    hewikt.fetch({lem for lem, b in all_lemmas if b and letters(lem) and uncovered(lem, b)})

    def root_of(lem, b):
        """manual > English Wiktionary > rule (trusted classes only) > Hebrew Wiktionary > None."""
        k = (letters(lem), b)
        if k in manual:
            return manual[k], "manual"
        if k in lex and lex[k]["root"]:
            return lex[k]["root"], "wiktionary"
        if rule_ok(lem, b):
            return derive_root(lem, b), "derived"
        r = hewikt.root(lem, b) if b in BINYAN.values() else None
        return (r, "hewiktionary") if r else (None, None)

    # the two dictionaries against each other, on verbs where both have a root (pages already in the cache)
    both = [(e["root"], hewikt.candidates(e["lemma"], b)) for (_, b), e in lex.items()
            if e["root"] and e["lemma"] in hewikt.pages]
    both = [(g, c) for g, c in both if c]

    # ---- lexicon
    analyses = defaultdict(set)  # spelling -> {(lemma, binyan, tense, person, gender, number)}
    lex_rows = []
    for (lem, b), e in sorted(lex.items()):
        root, src = root_of(e["lemma"], b)
        for c, (spellings, how) in sorted(e["cells"].items(), key=lambda x: str(x[0])):
            for s in sorted(spellings):
                analyses[s].add((lem, b) + c)
                lex_rows.append({"form": s, "spelling": how, "lemma": lem, "binyan": b, "tense": c[0], "person": c[1],
                                 "gender": c[2], "number": c[3], "root": root, "root_source": src,
                                 "root_class": root_class(root), "gloss": e["gloss"]})
    for r in lex_rows:
        a = analyses[r["form"]]
        r["n_analyses"], r["n_binyanim"] = len(a), len({x[1] for x in a})

    # ---- UD verbs in context
    ud_rows, missing_span = [], 0
    for tb, url in UD.items():
        for split in ("train", "dev", "test"):
            for sid, text, rows, spans in read_conllu(os.path.join(raw, os.path.basename(url.format(split)))):
                for c in rows:
                    if "-" in c[0] or "." in c[0] or c[3] != "VERB":
                        continue
                    f = ud_feats(c[5])
                    b, tense = f.get("HebBinyan"), ud_tense(f)
                    if b is None:  # יש/אין and a few others
                        continue
                    start, end, word = spans.get(int(c[0]), (None, None, None))
                    missing_span += start is None
                    h = word.find(c[1]) if word else -1
                    root, src = root_of(c[2], b)
                    a = analyses.get(c[1], set())
                    ud_rows.append({
                        "treebank": tb, "split": split, "sent_id": sid, "text": text,
                        "word": word, "start": start, "end": end, "form": c[1],
                        "host_start": start + h if h >= 0 else None, "host_end": start + h + len(c[1]) if h >= 0 else None,
                        "lemma": c[2], "binyan": b, "tense": tense,
                        "person": f.get("Person") if tense not in ("present", "infinitive") else None,
                        "gender": {"Masc": "m", "Fem": "f", "Fem,Masc": "m,f"}.get(f.get("Gender"), f.get("Gender")),
                        "number": {"Sing": "sg", "Plur": "pl"}.get(f.get("Number"), f.get("Number")),
                        "voice": f.get("Voice"), "feats": c[5],
                        "root": root, "root_source": src, "root_class": root_class(root),
                        "in_lexicon": (letters(c[2]), b) in lex,
                        "lex_n_analyses": len(a), "lex_n_binyanim": len({x[1] for x in a}),
                    })

    # ---- roots to check by hand: verbs no source gives a root for
    ud_count, ud_example, spelled = Counter(), {}, {k: e["lemma"] for k, e in lex.items()}
    for r in ud_rows:
        k = (letters(r["lemma"]), r["binyan"])
        ud_count[k] += 1
        ud_example.setdefault(k, r["text"])
        spelled.setdefault(k, r["lemma"])
    check = {}
    for k in set(ud_count) | set(lex):
        lem, b = k
        if b in BINYAN.values() and len(lem) >= 2 and (k in manual or root_of(spelled.get(k, lem), b)[0] is None):
            check[k] = derive_root(lem, b)
    with open(check_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["lemma", "binyan", "derived_root", "derived_class", "rule_accuracy_in_class",
                    "hewiktionary_candidates", "ud_tokens", "in_lexicon", "example", "correct_root"])
        for (lem, b), d in sorted(check.items(), key=lambda x: (-ud_count[x[0]], x[0])):
            c = root_class(d)
            w.writerow([spelled.get((lem, b), lem), b, d, c, f"{rule_acc.get(c, 0):.2f}",
                        " ".join(sorted(hewikt.candidates(spelled.get((lem, b), lem), b))), ud_count[(lem, b)],
                        (lem, b) in lex, ud_example.get((lem, b), ""), manual.get((lem, b)) or ""])

    for name, rows in (("lexicon.jsonl", lex_rows), ("ud_verbs.jsonl", ud_rows)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    roots = defaultdict(set)
    for r in ud_rows:
        if r["root"]:
            roots[r["root"]].add(r["binyan"])
    stats = {
        "ud_verbs": len(ud_rows), "ud_missing_span": missing_span,
        "ud_by_treebank": Counter(r["treebank"] for r in ud_rows),
        "ud_by_tense": Counter(r["tense"] for r in ud_rows),
        "ud_by_binyan": Counter(r["binyan"] for r in ud_rows),
        "ud_by_root_source": Counter(r["root_source"] for r in ud_rows),
        "ud_by_root_class": Counter(r["root_class"] for r in ud_rows),
        "ud_distinct_lemma_binyan": len(ud_count), "ud_distinct_roots": len(roots),
        "ud_roots_in_2plus_binyanim": sum(len(v) >= 2 for v in roots.values()),
        "ud_roots_in_3plus_binyanim": sum(len(v) >= 3 for v in roots.values()),
        "ud_ambiguous_form": sum(r["lex_n_analyses"] > 1 for r in ud_rows),
        "lexicon_rows": len(lex_rows), "lexicon_verbs": len(lex),
        "lexicon_verbs_with_wiktionary_root": sum(1 for e in lex.values() if e["root"]),
        "lexicon_by_spelling": Counter(r["spelling"] for r in lex_rows),
        "lexicon_ambiguous_forms": sum(len(a) > 1 for a in analyses.values()),
        "lexicon_forms_in_2plus_binyanim": sum(len({x[1] for x in a}) > 1 for a in analyses.values()),
        "rule_accuracy_by_class": {c: [round(rule_acc[c], 3), rule_n[c]] for c in sorted(rule_n)},
        "rule_trusted_classes": sorted(trusted),
        "ud_rooted_by_class": Counter(r["root_class"] for r in ud_rows if r["root"]),
        "ud_rooted_verbs_by_class": Counter(c for c, _ in {(r["root_class"], (r["lemma"], r["binyan"]))
                                                          for r in ud_rows if r["root"]}),
        "hewiktionary_vs_wiktionary": [sum(g in c for g, c in both), len(both)],
        "hewiktionary_pages_cached": len(hewikt.pages),
        "to_check": len(check), "to_check_filled": sum(1 for k in check if k in manual),
        "manual_roots": len(manual),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
