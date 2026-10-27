"""Builds site/docs from the RQ reports so MkDocs can render the RQ Explorer website.
The reports stay the single source: nothing in result-analysis/ is edited. This script copies each report, puts the
figure images above their captions, links every "Figure N" mention to its figure, and writes the home page."""
import re, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = HERE / "docs"
RES = ROOT / "result-analysis"
N_RQ = 7

RQ6_QUESTION = "Which Word of the Prompt Carries the Credit, the Organ Word, the Position Word, the Other Words or the Image?"
ORDINAL = {"a": "first", "b": "second", "c": "third", "d": "last"}


def rq6_caption(key):
    group, letter = key
    side = "left" if group == "1" else "right"
    return (f"Shapley value of the image, the organ word, the position word and the other words "
            f"for the {ORDINAL[letter]} four {side} organs of subject 14")


def figure_files(n):
    """{(number, letter): image path}. RQ3 keeps its organ-pair figures in figures_with_images."""
    d = RES / f"RQ{n}-result-analysis"
    out = {}
    for p in sorted((d / "figures").glob("F*.png")):
        m = re.match(r"F(\d+)([a-z]?)_", p.name)
        if m:
            out[(m.group(1), m.group(2))] = p
    extra = d / "figures_with_images"
    if extra.exists():
        for k, p in enumerate(sorted(extra.glob("*.png")), start=2):
            out[(str(k), "")] = p
    return out


def copy_assets(n, figs):
    dest = DOCS / "assets" / f"rq{n}"
    dest.mkdir(parents=True, exist_ok=True)
    web = {}
    for key, p in figs.items():
        shutil.copy2(p, dest / p.name)
        web[key] = f"assets/rq{n}/{p.name}"
    return web, []


def fig_html(src, num):
    return (f'<figure class="rqfig"><a class="figlink" href="{src}" data-fig="{num}">'
            f'<img src="{src}" alt="Figure {num}" loading="lazy"></a></figure>')


def link_mentions(text, seen, cap_ids):
    """Link 'Figure N', 'Figures A to B' and 'Figures A and B'. The first mention of each figure gets a back anchor."""
    def link(num):
        if num not in cap_ids:
            return None
        anchor = "" if num in seen else f'<a id="ref-fig{num}"></a>'
        seen.add(num)
        return f"{anchor}[{num}](#fig{num})"

    def one(m):
        r = link(m.group(1))
        return "Figure " + r if r else m.group(0)

    def two(m):
        a, b = link(m.group(1)), None
        if a is None:
            return m.group(0)
        b = link(m.group(3))
        return f"Figures {a}{m.group(2)}{b}" if b else f"Figures {a}{m.group(2)}{m.group(3)}"

    text = re.sub(r"Figures (\d+)( to | and )(\d+)", two, text)
    return re.sub(r"(?<![\[\w])Figure (\d+)(?![\w\]])", one, text)


def process_report(text, web):
    lines = text.replace("\r\n", "\n").split("\n")
    cap_ids = {m.group(1) for l in lines if (m := re.match(r"\*\*Figure (\d+)\.", l))}
    out, seen = [], set()
    in_conclusion, red_done = False, False
    for l in lines:
        mc = re.match(r"\*\*Figure (\d+)\. (.*)\*\*\s*$", l)
        if mc:
            num, title = mc.group(1), mc.group(2)
            src = web.get((num, ""))
            if src:
                out += [fig_html(src, num), ""]
            out += [f'<p class="cap" id="fig{num}"><strong>Figure {num}. {title}</strong> '
                    f'<!--BACK{num}--></p>', ""]
            continue
        if l.startswith("## "):
            in_conclusion = l.strip().lower() == "## conclusion"
        elif l.startswith("*Note.*"):
            l = l.replace("*Note.*", '<span class="red">Note.</span>', 1)
        if in_conclusion and not red_done and l and not l.startswith("#"):
            m = re.search(r"\d+\.\d+|\d+%", l)
            if m:
                l = l[:m.start()] + f'<span class="red">{m.group(0)}</span>' + l[m.end():]
                red_done = True
        if not l.startswith("#"):
            l = link_mentions(l, seen, cap_ids)
        out.append(l)
    page = "\n".join(out)
    for num in cap_ids:
        back = f' <a class="back" href="#ref-fig{num}">back to the text</a>' if num in seen else ""
        page = page.replace(f"<!--BACK{num}-->", back)
    return page


def move_definitions_last(md):
    """Question, hypothesis, tables, figures, key findings and conclusion first, then the definitions."""
    parts = re.split(r"(?m)^(?=## )", md)
    defs = [p for p in parts if p.startswith("## Definition")]
    rest = [p for p in parts if not p.startswith("## Definition")]
    rest = [p.rstrip("\n") + "\n\n" for p in rest]
    return "".join(rest + [d.rstrip("\n") + "\n" for d in defs])


def downloads(tabs):
    if not tabs:
        return ""
    return "\n\n## Download the tables\n\n" + "\n".join(f"- [{name}]({href})" for name, href in tabs) + "\n"


def first_finding(text):
    m = re.search(r"## Key findings\s+1\.\s+\*\*(.+?)\*\*", text, re.S)
    return m.group(1).strip() if m else ""


def question(text):
    m = re.match(r"# RQ\d+: (.*)", text)
    return m.group(1).strip() if m else ""


def placeholder_page(web):
    body = [f"# RQ6: {RQ6_QUESTION}", "",
            '<p><span class="red">The report for this research question is not written yet.</span> The figures below are ready. '
            "The tables, the key findings and the conclusion will be added when the analysis is done.</p>", "",
            "The credit of each part of the prompt is its Shapley value, in Dice units. Blue raises the Dice score and red lowers it.", ""]
    for key, src in sorted(web.items()):
        num = key[0] + key[1]
        body += [fig_html(src, num), "",
                 f'<p class="cap" id="fig{num}"><strong>Figure {num}. {rq6_caption(key)}</strong></p>', ""]
    return "\n".join(body)


def short(q, n=64):
    return q if len(q) <= n else q[:n].rsplit(" ", 1)[0] + "..."


def main():
    import json
    import render
    OUT = HERE / "_site"
    if DOCS.exists():
        shutil.rmtree(DOCS)
    if OUT.exists():
        for child in OUT.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    DOCS.mkdir(parents=True)
    rows, pages = [], {}
    for n in range(1, N_RQ + 1):
        web, tabs = copy_assets(n, figure_files(n))
        rp = RES / f"RQ{n}-result-analysis" / f"RQ{n}_report.md"
        if rp.exists():
            raw = rp.read_text(encoding="utf-8")
            md = move_definitions_last(process_report(raw, web))
            q, key, ready = question(raw), first_finding(raw), True
        else:
            md = placeholder_page(web); q, key, ready = RQ6_QUESTION, "", False
        pages[f"rq{n}"] = (f"RQ{n}", md, q)
        rows.append((n, q, key, len(web), 0, ready))
    nav = [("index", "Home", "")] + [(f"rq{n}", f"RQ{n}", short(q)) for n, q, *_ in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    keys = [f"rq{n}" for n in range(1, N_RQ + 1)]
    for key, (label, md, q) in pages.items():
        body, _ = render.md_to_html(md)
        k = keys.index(key) if key in keys else -1
        prev = (keys[k - 1], keys[k - 1].upper()) if k > 0 else None
        nxt = (keys[k + 1], keys[k + 1].upper()) if 0 <= k < len(keys) - 1 else None
        (OUT / f"{key}.html").write_text(render.page(label, body, nav, key, (), prev, nxt), encoding="utf-8")
        plain = re.sub(r"<[^>]+>", " ", body)
        index.append({"key": key, "title": f"{label} {q}".lower(), "text": re.sub(r"\s+", " ", plain).lower()})
    (OUT / "index.html").write_text(render.page("Home", render.home(rows), nav, "index", wide=True), encoding="utf-8")
    shutil.copytree(DOCS / "assets", OUT / "assets", dirs_exist_ok=True)
    (OUT / "assets" / "extra.css").write_text(render.EXTRA_CSS, encoding="utf-8")
    (OUT / "assets" / "site.js").write_text(render.SITE_JS, encoding="utf-8")
    shutil.copy2(HERE / "assets" / "figures.js", OUT / "assets" / "figures.js")
    (OUT / "assets" / "search.json").write_text(json.dumps(index), encoding="utf-8")
    shutil.rmtree(DOCS)
    print("built", len(pages) + 1, "pages into", OUT)


if __name__ == "__main__":
    main()
