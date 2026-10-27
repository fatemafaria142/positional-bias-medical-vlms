# RQ Explorer

A free static website for RQ1 to RQ7. The sidebar lists the research questions, and each page shows the full report with
its tables and figures. Hover over a figure to see it at full size, or click it to move between the figures of the page
(Previous and Next buttons, the left and right arrow keys, Escape to close).

## How it works
- The reports in `result-analysis/RQn-result-analysis/RQn_report.md` are the only source. They are never edited.
- `build_site.py` copies them into `site/docs/`, puts each figure image above its caption, links every "Figure N" in the text
  to its figure, and writes the home page and the table download lists.
- `render.py` turns the markdown into HTML pages styled with Tailwind CSS (loaded from the Tailwind CDN with the typography
  plugin), and writes the website to `site/_site/`. It has a sticky sidebar, search, an outline of each page, and Previous and Next links.
- `site/_site/` is generated and is not committed.

## Run it on your computer
```
pip install markdown
python site/build_site.py
python -m http.server 8000 --directory site/_site
```
Then open http://127.0.0.1:8000. The page needs internet once to load Tailwind and the font.

## Publish it free on GitHub Pages
1. Push this repository to GitHub. The workflow `.github/workflows/pages.yml` builds the site on every push.
2. In the repository, open Settings, then Pages, and set the source to **GitHub Actions**.
3. In Settings, then Environments, then `github-pages`, make sure the branch you push from is allowed to deploy
   (or merge to `main`).
4. The site address is shown at the end of the workflow run, usually `https://<user>.github.io/<repository>/`.

The site is public only if the repository is public.

## Update it
Change a report or a figure, push, and the site rebuilds. A new RQ only needs a bump of `N_RQ` in `build_site.py`.
