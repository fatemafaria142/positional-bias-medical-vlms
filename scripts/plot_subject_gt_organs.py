"""Ground-truth organ masks of one OASIS subject (the 4 organs used in rows 13-16), for slides.
Usage: python scripts/plot_subject_gt_organs.py [subject_id] [person]
Output: local_plots_groundedness/subject_gt/<subject>_overview.png and <subject>_per_organ.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sid = sys.argv[1] if len(sys.argv) > 1 else "OASIS_OAS1_0014_MR1"
person = sys.argv[2] if len(sys.argv) > 2 else "faria"
vol = REPO / person / "samples" / "neurite_oasis" / "volumes"
img = np.asarray(nib.load(vol / f"{sid}_rot000_image.nii.gz").dataobj, dtype=np.float32)
lab = np.asarray(nib.load(vol / f"{sid}_rot000_label.nii.gz").dataobj).astype(int)   # axes: L, I, A

ORGANS = {"left cerebral white matter": (1, "#2563eb"), "right cerebral white matter": (20, "#16a34a"),
          "left cerebellum cortex": (6, "#f59e0b"), "right cerebellum cortex": (25, "#dc2626")}
masks = {o: lab == i for o, (i, _) in ORGANS.items()}
def centre(m):                               # slice indices (L, I, A) through the centre of a mask
    return [int(round(v.mean())) for v in np.nonzero(m)]

GROUPS = {"Cerebral white matter": ["left cerebral white matter", "right cerebral white matter"],
          "Cerebellum cortex": ["left cerebellum cortex", "right cerebellum cortex"]}
lo, hi = np.percentile(img[img > 0], [1, 99.5])

def view(a, kind, k):                       # radiological display: patient left on image right, superior up, anterior up
    if kind == "coronal":
        return a[:, :, k].T
    return a[:, k, :].T[::-1]

def slice_of(kind, c):                       # c = (x, y, z) centre of the structure
    return c[2] if kind == "coronal" else c[1]

def rgba(m, color):
    c = np.zeros(m.shape + (4,)); c[m] = matplotlib.colors.to_rgba(color, 0.65); return c

out = REPO / "local_plots_groundedness" / "subject_gt"; out.mkdir(parents=True, exist_ok=True)

KINDS = [("Coronal", "coronal"), ("Axial", "axial")]
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for r, (gname, organs) in enumerate(GROUPS.items()):
    c = centre(np.any([masks[o] for o in organs], axis=0))
    for ci, (kname, kind) in enumerate(KINDS):
        ax = axes[r, ci]; k = slice_of(kind, c)
        ax.imshow(view(img, kind, k), cmap="gray", vmin=lo, vmax=hi)
        for o in organs:
            ax.imshow(rgba(view(masks[o], kind, k), ORGANS[o][1]))
        ax.set_title(f"{gname} - {kname}", fontsize=13); ax.axis("off")
fig.legend(handles=[Patch(color=c, label=o) for o, (_, c) in ORGANS.items()], loc="lower center", ncol=4, frameon=False, fontsize=11)
fig.text(0.5, 0.965, "Ground-truth masks (patient left is on the image right)", ha="center", fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 0.955)); fig.savefig(out / f"{sid}_overview.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
for j, (o, (_, col)) in enumerate(ORGANS.items()):
    c = centre(masks[o])
    for i, (kname, kind) in enumerate(KINDS):
        ax = axes[i, j]; k = slice_of(kind, c)
        ax.imshow(view(img, kind, k), cmap="gray", vmin=lo, vmax=hi)
        ax.imshow(rgba(view(masks[o], kind, k), col)); ax.axis("off")
        if i == 0: ax.set_title(f"{o}\n{masks[o].sum():,} voxels", fontsize=12)
        if j == 0: ax.text(-0.02, 0.5, kname, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=13)
fig.tight_layout(); fig.savefig(out / f"{sid}_per_organ.png", dpi=200); plt.close(fig)
print("wrote", out)
