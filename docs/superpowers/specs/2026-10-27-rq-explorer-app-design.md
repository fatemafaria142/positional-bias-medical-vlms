# RQ Explorer, a free public website for RQ1 to RQ7

## Goal
A free, public website with the seven research questions in a sidebar. Clicking one opens its full report with
tables and figures. Hosted on GitHub Pages, built from the existing report files.

## Decisions made with the user
- Public website on GitHub Pages (free).
- Each RQ page shows the full report on one scrolling page.
- Figures are linked and can be enlarged to the full page.
- Text uses bold black and red where needed.

## Architecture
- **Tool:** MkDocs with the Material theme.
- **Source of truth:** the existing files `result-analysis/RQn-result-analysis/RQn_report.md`,
  `RQ1_to_RQ7_Methodology.md` and `result-analysis/RQ_Master_Analysis.md`. They are not moved or edited.
- **New files, all under `site/`:** `mkdocs.yml`, `build_site.py`, `assets/extra.css`, `assets/figures.js`, and
  `.github/workflows/pages.yml`.
- **Build:** `build_site.py` copies each report into `site/docs/`, adds the figure images and the table download
  links, then MkDocs builds the static site. The workflow runs on every push to the branch and publishes to Pages.

## Pages and navigation
- Sidebar: Home, RQ1, RQ2, RQ3, RQ4, RQ5, RQ6, RQ7, Methodology, Master analysis. Search is on.
- Home: a table of the 7 RQs with the question, the key result and a link to each page.
- RQ page: the report as written (question, hypothesis, definitions, tables, figures, key findings, conclusion).
- RQ6 has no report yet, so its page shows its figures and a note in red that the report is not written yet.

## Figures
- The reports mention "Figure N" but do not embed images. The build script inserts the image from the RQ's
  `figures/` folder above each "Figure N." caption, matched by figure number (F1_..., F2_...).
- **Linked:** each "Figure N" mention in the text becomes a link that jumps to that figure. Each figure caption
  links back to where it is first mentioned.
- **Hover to enlarge (desktop):** hovering a figure shows it at full window size. Moving the mouse away closes it.
- **Click to open (also touch):** clicking opens a full-page viewer with Previous and Next buttons, the left and
  right arrow keys, and Escape to close, moving through the figures of the same RQ.

## Text colours
- Bold black: headings, table headers, the lead sentence of each key finding, and the labels "What it shows",
  "What to notice" and "Why it matters".
- Bold red: the "Conclusion" label, caveats and warnings, and the one headline number of each RQ's conclusion.
- No gray text anywhere. Plain body text is black.

## Tables
Shown as normal tables. Each table has a download link to its CSV in `result-analysis/RQn-result-analysis/tables/`.
The large result CSVs in `results/` are not part of the site.

## Hosting
GitHub Pages with GitHub Actions as the source. The user switches Pages on once in the repository settings.
The site is public only if the repository is public.

## Testing
- Build locally and check that every RQ page loads and every figure image resolves.
- Check that each "Figure N" link and each caption link goes to the right place.
- Check hover, click, arrow-key and Escape behaviour on the figure viewer.
- Check the layout at phone width.

## Out of scope
RQ6 tables and report, new analyses, and any change to the report text.
