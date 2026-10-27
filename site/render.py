"""HTML templates for the RQ Explorer, styled with Tailwind CSS (loaded from the Tailwind CDN, with the typography plugin)."""
import html, json, re
import markdown

TAILWIND = "https://cdn.tailwindcss.com/3.4.17?plugins=typography"

TAILWIND_CONFIG = """
tailwind.config = { theme: { extend: { colors: { ink: '#000000', brand: '#d00000' }, fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'sans-serif'] } } } }
"""

PROSE = ("prose prose-lg max-w-none text-black prose-headings:text-black prose-headings:font-extrabold prose-headings:scroll-mt-24 "
         "prose-h1:text-3xl sm:prose-h1:text-4xl prose-h2:mt-12 prose-h2:border-b-4 prose-h2:border-black prose-h2:pb-2 prose-strong:text-black prose-strong:font-extrabold "
         "prose-a:text-black prose-a:font-semibold prose-a:underline prose-a:decoration-2 hover:prose-a:text-brand "
         "prose-li:text-black prose-p:text-black prose-code:text-black prose-th:text-black prose-td:text-black")


def md_to_html(text):
    md = markdown.Markdown(extensions=["tables", "attr_list", "md_in_html", "toc", "sane_lists"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(text)
    body = re.sub(r"<table>", '<div class="tablewrap"><table>', body)
    body = body.replace("</table>", "</table></div>")
    toc = [(m.group(1), re.sub(r"<[^>]+>", "", m.group(2))) for m in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body)]
    return body, toc


def sidebar(nav, current):
    items = []
    for key, label, sub in nav:
        active = key == current
        cls = ("border-l-4 border-brand bg-red-50 text-brand font-extrabold" if active
               else "border-l-4 border-transparent text-black hover:bg-gray-100 font-semibold")
        items.append(f'<a href="{key}.html" data-nav="{key}" class="block px-4 py-2 rounded-r-lg {cls}">'
                     f'<span class="block text-sm">{label}</span>'
                     + (f'<span class="block text-xs font-normal text-black leading-snug">{html.escape(sub)}</span>' if sub else "")
                     + "</a>")
    return "\n".join(items)


def page(title, body_html, nav, current, toc=(), prev=None, nxt=None, wide=False):
    toc_html = ""
    if toc:
        toc_html = ('<aside class="hidden xl:block w-60 shrink-0"><div class="sticky top-24"><p class="text-sm font-extrabold uppercase tracking-wide text-black mb-3">On this page</p>'
                    '<ul class="space-y-2 border-l-2 border-black pl-4 text-sm">'
                    + "".join(f'<li><a class="text-black font-semibold hover:text-brand" href="#{i}">{html.escape(t)}</a></li>' for i, t in toc)
                    + "</ul></div></aside>")
    pn = ""
    if prev or nxt:
        def card(side, tgt):
            if not tgt: return "<div></div>"
            return (f'<a href="{tgt[0]}.html" class="group block rounded-xl border-2 border-black p-4 hover:border-brand no-underline {"text-right" if side == "next" else ""}">'
                    f'<span class="block text-xs font-extrabold uppercase text-black group-hover:text-brand">{"Next" if side == "next" else "Previous"}</span>'
                    f'<span class="block text-lg font-extrabold text-black group-hover:text-brand">{html.escape(tgt[1])}</span></a>')
        pn = f'<nav class="mt-16 grid grid-cols-1 sm:grid-cols-2 gap-4">{card("prev", prev)}{card("next", nxt)}</nav>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} - RQ Explorer</title>
<script src="{TAILWIND}"></script>
<script>{TAILWIND_CONFIG}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect width=%2732%27 height=%2732%27 rx=%276%27/%3E%3Ctext x=%2716%27 y=%2721%27 font-size=%2714%27 font-weight=%27bold%27 text-anchor=%27middle%27 fill=%27white%27%3ERQ%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/extra.css">
</head>
<body class="bg-white text-black font-sans antialiased">
<header class="sticky top-0 z-40 bg-white border-b-4 border-black">
  <div class="mx-auto flex max-w-screen-2xl items-center gap-3 px-4 py-3">
    <button id="menu" class="lg:hidden rounded-lg border-2 border-black px-3 py-1 font-extrabold" aria-label="Menu">Menu</button>
    <a href="index.html" class="flex items-center gap-2 text-xl font-extrabold text-black no-underline"><span class="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-black text-white text-sm">RQ</span><span class="hidden sm:inline">RQ Explorer</span></a>
    <span class="hidden sm:inline text-sm font-semibold text-black">How VoxTell handles left and right on brain MRI</span>
    <div class="ml-auto w-40 sm:w-72"><input id="search" type="search" placeholder="Search the reports" class="w-full rounded-lg border-2 border-black px-3 py-1.5 text-sm font-semibold text-black placeholder-black focus:outline-none focus:border-brand"></div>
  </div>
</header>
<div class="mx-auto flex max-w-screen-2xl gap-8 px-4">
  <aside id="side" class="fixed inset-y-0 left-0 z-30 hidden w-72 overflow-y-auto bg-white border-r-4 border-black pt-20 pb-8 lg:sticky lg:top-16 lg:z-0 lg:block lg:h-[calc(100vh-4rem)] lg:w-72 lg:shrink-0 lg:border-r-0 lg:pt-6">
    <nav class="space-y-1">{sidebar(nav, current)}</nav>
    <p id="nohits" class="hidden px-4 pt-4 text-sm font-extrabold text-brand">No page matches.</p>
  </aside>
  <main class="min-w-0 flex-1 py-8 {"" if wide else "max-w-4xl"}">
    <article class="{PROSE}">{body_html}</article>
    {pn}
    <footer class="mt-16 border-t-4 border-black py-6 text-sm font-semibold text-black">RQ Explorer. Built from the RQ reports in this repository.</footer>
  </main>
  {toc_html}
</div>
<div id="side-backdrop" class="fixed inset-0 z-20 hidden bg-black/40 lg:hidden"></div>
<script src="assets/figures.js"></script>
<script src="assets/site.js"></script>
</body></html>"""


CARD = """<a href="rq{n}.html" class="group flex flex-col rounded-2xl border-2 border-black bg-white p-6 no-underline shadow-[6px_6px_0_0_#000] transition hover:-translate-y-1 hover:border-brand hover:shadow-[6px_6px_0_0_#d00000]">
<span class="text-4xl font-extrabold text-black group-hover:text-brand">RQ{n}</span>
<span class="mt-3 text-base font-extrabold leading-snug text-black">{question}</span>
<span class="mt-4 text-xs font-extrabold uppercase tracking-wide text-black">Key result</span>
<span class="text-sm font-semibold leading-snug text-black">{key}</span>
<span class="mt-5 flex flex-wrap gap-2">{badges}</span></a>"""


def home(rows):
    cards = []
    for n, q, key, nfig, ntab, ready in rows:
        badges = [f'<span class="rounded-full border-2 border-black px-3 py-0.5 text-xs font-extrabold text-black">{nfig} figure{"s" if nfig != 1 else ""}</span>']
        if ntab: badges.append(f'<span class="rounded-full border-2 border-black px-3 py-0.5 text-xs font-extrabold text-black">{ntab} table file{"s" if ntab != 1 else ""}</span>')
        if not ready: badges.append('<span class="rounded-full bg-brand px-3 py-0.5 text-xs font-extrabold text-white">Report not written yet</span>')
        cards.append(CARD.format(n=n, question=html.escape(q), key=html.escape(key) if key else '<span class="text-brand font-extrabold">Report not written yet</span>', badges="".join(badges)))
    return f"""<section class="not-prose">
<p class="text-sm font-extrabold uppercase tracking-widest text-brand">Research questions</p>
<h1 class="mt-2 text-4xl font-extrabold leading-tight text-black sm:text-5xl">Does VoxTell tell <span class="text-brand">left</span> from <span class="text-brand">right</span>?</h1>
<p class="mt-5 max-w-3xl text-lg font-semibold leading-relaxed text-black">Seven research questions on how a text-promptable 3D segmentation model handles the words left and right on brain MRI.
Pick a question to read its full report, with tables and figures. Hover over a figure to see it at full size, or click it to move between the figures of the page.</p>
<div class="mt-10 grid gap-8 md:grid-cols-2 2xl:grid-cols-3">{"".join(cards)}</div>
<div class="mt-12 flex flex-wrap gap-4">
<a href="methodology.html" class="rounded-xl border-2 border-black px-5 py-3 font-extrabold text-black no-underline hover:border-brand hover:text-brand">Methodology</a>
<a href="master.html" class="rounded-xl border-2 border-black px-5 py-3 font-extrabold text-black no-underline hover:border-brand hover:text-brand">Master analysis</a></div></section>"""


SITE_JS = """/* Sidebar search over the page text, and the mobile menu. */
(function () {
  var side = document.getElementById('side'), back = document.getElementById('side-backdrop');
  function toggle(open) { side.classList.toggle('hidden', !open); back.classList.toggle('hidden', !open); }
  var menu = document.getElementById('menu');
  if (menu) menu.onclick = function () { toggle(side.classList.contains('hidden')); };
  if (back) back.onclick = function () { toggle(false); };
  var input = document.getElementById('search'), idx = null;
  if (!input) return;
  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase();
    var go = function () {
      var shown = 0;
      document.querySelectorAll('[data-nav]').forEach(function (a) {
        var e = idx.find(function (x) { return x.key === a.getAttribute('data-nav'); });
        var ok = !q || (e && (e.title + ' ' + e.text).indexOf(q) >= 0);
        a.style.display = ok ? '' : 'none'; if (ok) shown++;
      });
      document.getElementById('nohits').classList.toggle('hidden', shown > 0);
      if (q && side.classList.contains('hidden')) toggle(true);
    };
    if (idx) return go();
    fetch('assets/search.json').then(function (r) { return r.json(); }).then(function (d) { idx = d; go(); });
  });
})();
"""

EXTRA_CSS = """/* Small pieces Tailwind classes do not cover. Black text, bold black emphasis, red only where needed. */
.red { color: #d00000; font-weight: 800; }
.tablewrap { overflow-x: auto; margin: 1.5rem 0; border: 2px solid #000000; border-radius: 0.5rem; }
.tablewrap table { margin: 0; width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.tablewrap th { background: #ffffff; color: #000000; font-weight: 800; border-bottom: 3px solid #000000; padding: 0.6rem 0.8rem; text-align: left; }
.tablewrap td { color: #000000; border-top: 1px solid #000000; padding: 0.55rem 0.8rem; }
.tablewrap tbody tr:nth-child(even) { background: #fff5f5; }
.prose a, .prose li, .prose p { overflow-wrap: anywhere; }
.prose li::marker { color: #000000; }
.prose ol > li::marker { font-weight: 800; }
.prose h2#conclusion { color: #d00000; border-color: #d00000; }
.rqfig { margin: 1.8rem 0 0.4rem; text-align: center; }
.rqfig img { max-width: 100%; height: auto; border: 2px solid #000000; border-radius: 0.5rem; cursor: zoom-in; background: #ffffff; margin: 0 auto; }
.cap { margin-top: 0.3rem; }
.cap .back { font-size: 0.8em; margin-left: 0.5rem; font-weight: 600; }
.cap:target strong { color: #d00000; }
:target { scroll-margin-top: 5rem; }
#fig-hover, #fig-viewer { position: fixed; inset: 0; z-index: 1000; display: none; background: #ffffff; }
#fig-hover { pointer-events: none; }
#fig-hover.on, #fig-viewer.on { display: flex; align-items: center; justify-content: center; }
#fig-hover img, #fig-viewer img { max-width: 97vw; max-height: 94vh; object-fit: contain; border: 2px solid #000000; background: #ffffff; }
#fig-viewer .bar { position: fixed; left: 0; right: 0; top: 0; display: flex; justify-content: space-between; align-items: center;
  padding: 0.5rem 1rem; background: #ffffff; border-bottom: 4px solid #000000; font-weight: 800; color: #000000; }
#fig-viewer button { font: inherit; font-weight: 800; color: #000000; background: #ffffff; border: 2px solid #000000; border-radius: 0.5rem; padding: 0.2rem 0.9rem; cursor: pointer; margin-left: 0.4rem; }
#fig-viewer button:hover { color: #ffffff; background: #d00000; border-color: #d00000; }
#fig-viewer img { margin-top: 2.8rem; max-height: 90vh; }
@media (hover: none) { #fig-hover { display: none !important; } }
"""
