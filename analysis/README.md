# analysis/ — figures and write-ups for the Axis II results

Post-processing only: these scripts read the finished experiment outputs in `../results/` and produce figures.
The experiment code itself (prompt catalogue, VoxTell engine, Kaggle runners) stays in `../scripts/`.

| File | Produces |
|---|---|
| `axis2_plot_style.py` | Shared palette, plot helpers, MRI/mask loading, the 3 example subjects (`EXAMPLE_SUBJECTS`) |
| `plot_axis2_exp1.py` | `results/results_axis2_exp1/plots/` |
| `plot_axis2_exp2_3.py` | `results/results_axis2_exp2_3/plots/` |
| `plot_axis2_exp4.py` | `results/results_axis2_exp4/plots/` |
| `plot_axis2_exp5.py` | `results/results_axis2_exp5/plots/` |
| `plot_axis2_experimental_plots.py` | `results/*/experimental_plot/` — 20 per-subject figures + `README.md` per experiment |

The `explanation.md` files in each results folder were written from these figures and numbers.
The website (`../website/build_site.py`) renders those markdown files.

## Run

From the repository root (`.venv` has matplotlib, pandas, nibabel, scipy, scikit-learn):

```bash
.venv/bin/python analysis/plot_axis2_exp1.py
.venv/bin/python analysis/plot_axis2_exp2_3.py
.venv/bin/python analysis/plot_axis2_exp4.py
.venv/bin/python analysis/plot_axis2_exp5.py
.venv/bin/python analysis/plot_axis2_experimental_plots.py
```

MRI example overlays need the local volumes in `data/neurite_oasis_100/volumes/` and the packed masks in
`results/*/<exp>/masks/` (both git-ignored); without them those figures are skipped.
