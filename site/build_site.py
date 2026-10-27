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
    tabs = []
    tdir = RES / f"RQ{n}-result-analysis" / "tables"
    if tdir.exists():
        for p in sorted(tdir.glob("*.csv")):
            (dest / "tables").mkdir(exist_ok=True)
            shutil.copy2(p, dest / "tables" / p.name)
            tabs.append((p.name, f"assets/rq{n}/tables/{p.name}"))
    return web, tabs


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


def main():
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True)
    rows = []
    for n in range(1, N_RQ + 1):
        web, tabs = copy_assets(n, figure_files(n))
        rp = RES / f"RQ{n}-result-analysis" / f"RQ{n}_report.md"
        if rp.exists():
            raw = rp.read_text(encoding="utf-8")
            page = process_report(raw, web) + downloads(tabs)
            rows.append((n, question(raw), first_finding(raw)))
        else:
            page = placeholder_page(web)
            rows.append((n, RQ6_QUESTION, ""))
        (DOCS / f"rq{n}.md").write_text(page, encoding="utf-8")
    shutil.copy2(ROOT / "RQ1_to_RQ7_Methodology.md", DOCS / "methodology.md")
    shutil.copy2(RES / "RQ_Master_Analysis.md", DOCS / "master.md")
    home = ["# RQ Explorer", "",
            "Seven research questions on how VoxTell, a text-promptable 3D segmentation model, handles **left and right** on brain MRI. "
            "Pick a question in the sidebar or in the table below. Every page has the full report with its tables and figures. "
            "Hover over a figure to see it at full size, or click it to move between the figures of the page.", "",
            "| Question | What it asks | Key result |", "|---|---|---|"]
    for n, q, k in rows:
        key = k if k else '<span class="red">Report not written yet</span>'
        home.append(f"| **[RQ{n}](rq{n}.md)** | {q} | {key} |")
    home += ["", "Also see the [methodology](methodology.md) and the [master analysis](master.md)."]
    (DOCS / "index.md").write_text("\n".join(home), encoding="utf-8")
    shutil.copytree(HERE / "assets", DOCS / "assets", dirs_exist_ok=True)
    print("built", len(rows), "RQ pages into", DOCS)


if __name__ == "__main__":
    main()
