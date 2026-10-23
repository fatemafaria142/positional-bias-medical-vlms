"""
Shared plotting style + helpers for the Axis II (VoxTell / Neurite-OASIS) result figures.

Palette follows the reference deck ("LIME, SHAP, Exact Shapley Attribution, Embedding, and
Attention Interventions on OASIS and AMOS22 dataset for VoxTell"): off-white background,
purple for position words, teal for organ words, gray for function words / neutral prompts.

Volumes and packed masks are handled in the same (S, A, R) array order the experiment engine
uses (`nib.as_closest_canonical(...)` followed by `.transpose(2, 1, 0)`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data" / "neurite_oasis_100" / "volumes"

BG = "#FFFCF8"
INK = "#2B2B2B"
GRID = "#E6E1DA"
POSITION = "#7B2FF7"   # position / side word
ORGAN = "#00A6A6"      # organ word
FUNCTION = "#9E9E9E"   # function words / neutral

LEFT = "#7B2FF7"
RIGHT = "#F28C28"
BASE = "#8C8C8C"
CONTRA = "#D62839"
GOOD = "#2A9D8F"
BAD = "#D62839"

VARIANT_COLORS: Dict[str, str] = {
    "base": BASE,
    "lateralized": ORGAN,
    "synonym": POSITION,
    "paraphrase": "#3A86FF",
    "location_based": "#FF9F1C",
    "contradictory": CONTRA,
    "absent_organ_negative": "#6D597A",
}

SETTING_COLORS: Dict[str, str] = {
    "both_on": GOOD,
    "image_only": "#3A86FF",
    "text_only": "#FF9F1C",
    "both_off": BAD,
}

# Example subjects for mask figures, chosen by Exp I mean Dice (canonical base/left/right prompts):
# median subject, lower-quartile subject, and the hardest subject of the 100.
EXAMPLE_SUBJECTS = [
    ("OASIS_OAS1_0086_MR1", "typical case (median subject)"),
    ("OASIS_OAS1_0001_MR1", "lower-quartile subject"),
    ("OASIS_OAS1_0031_MR1", "hardest subject"),
]


def subj_tag(subject_id: str) -> str:
    return subject_id.replace("OASIS_OAS1_", "").replace("_MR1", "")


CORE7 = [
    "hippocampus", "thalamus", "caudate", "putamen",
    "cerebral white matter", "cerebellum white matter", "cerebellum cortex",
]
SHORT_NAMES = {
    "hippocampus": "Hippocampus",
    "thalamus": "Thalamus",
    "caudate": "Caudate",
    "putamen": "Putamen",
    "cerebral white matter": "Cerebral WM",
    "cerebellum white matter": "Cerebellum WM",
    "cerebellum cortex": "Cerebellum ctx",
}
LABEL_PAIRS: Dict[str, Tuple[int, int]] = {
    "hippocampus": (14, 30),
    "thalamus": (7, 26),
    "caudate": (8, 27),
    "putamen": (9, 28),
    "cerebral white matter": (1, 20),
    "cerebellum white matter": (5, 24),
    "cerebellum cortex": (6, 25),
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": "#BDB6AC",
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_ROOT)}")
    return path


def suptitle(fig: plt.Figure, title: str, subtitle: Optional[str] = None) -> None:
    h = fig.get_figheight()
    n_lines = subtitle.count("\n") + 1 if subtitle else 0
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, x=0.01, ha="left", va="bottom",
                 y=1 + (0.24 + 0.18 * n_lines if subtitle else 0.12) / h)
    if subtitle:
        fig.text(0.01, 1 + 0.12 / h, subtitle, fontsize=10, color="#5C5C5C", ha="left", va="bottom")


def ci95(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return 0.0
    return 1.96 * v.std(ddof=1) / np.sqrt(v.size)


def bar_labels(ax, bars, fmt: str = "{:.2f}", fontsize: float = 8, inside: bool = False,
               errs: Optional[Sequence[float]] = None) -> None:
    for i, b in enumerate(bars):
        h = b.get_height()
        if np.isnan(h):
            continue
        y = h / 2 if inside else h + (errs[i] if errs is not None and not np.isnan(errs[i]) else 0)
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(h), ha="center",
                va="center" if inside else "bottom", fontsize=fontsize, color=INK)


def word_chips(ax, x: float, y: float, words: Iterable[Tuple[str, str]], fontsize: float = 9) -> float:
    """Draw prompt words as colored chips (deck style). `words` = [(text, kind)], kind in position/organ/function/contra."""
    colors = {"position": POSITION, "organ": ORGAN, "function": FUNCTION, "contra": CONTRA}
    renderer = ax.figure.canvas.get_renderer()
    for text, kind in words:
        t = ax.text(x, y, text, fontsize=fontsize, color="white" if kind != "function" else INK,
                    va="center", ha="left", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.25", fc=colors.get(kind, FUNCTION) if kind != "function" else "#ECE8E2",
                              ec="none"))
        bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        x = bb.x1 + 0.012
    return x


def tokenize_prompt(prompt: str, target: str) -> list:
    """Split a prompt into (word, kind) chips: side words purple, organ words teal, rest gray."""
    side_words = {"left", "right", "l.", "r.", "left-sided", "right-sided"}
    organ_words = set(target.lower().split()) | {
        "hippocampal", "thalamic", "cerebellar", "cerebral", "striatal", "lentiform", "nucleus",
        "formation", "cortex", "matter", "white", "caudate", "putamen", "thalamus", "hippocampus",
    }
    out = []
    for w in prompt.split():
        if w.endswith(".") and w.lower() not in ("l.", "r."):
            w = w[:-1]
        lw = w.lower().strip(",'")
        if not lw:
            continue
        if lw in side_words or lw.startswith("left") or lw.startswith("right"):
            out.append((w, "position"))
        elif lw in organ_words:
            out.append((w, "organ"))
        else:
            out.append((w, "function"))
    return out


# --------------------------------------------------------------------------------------------
# Volume / mask helpers (S, A, R order, same as the engine)
# --------------------------------------------------------------------------------------------
def load_sar(path: Path) -> np.ndarray:
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.ascontiguousarray(np.asarray(img.dataobj).transpose(2, 1, 0))


def load_subject(subject_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    img_p = DATA_DIR / f"{subject_id}_image.nii.gz"
    lab_p = DATA_DIR / f"{subject_id}_label.nii.gz"
    if not (img_p.exists() and lab_p.exists()):
        return None
    return load_sar(img_p).astype(np.float32), load_sar(lab_p).astype(np.int32)


def unpack(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(packed, count=n).astype(bool).reshape(tuple(int(s) for s in shape))


def norm_img(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 1)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def coronal(vol: np.ndarray, a_idx: int) -> np.ndarray:
    """(S, A, R) -> coronal slice (S up, R to the right of the image = patient's right)."""
    return vol[::-1, a_idx, :]


def axial(vol: np.ndarray, s_idx: int) -> np.ndarray:
    """(S, A, R) -> axial slice (anterior up, patient's right on the image's right)."""
    return vol[s_idx, ::-1, :]


def overlay(ax, base: np.ndarray, masks: Sequence[Tuple[np.ndarray, str]], alpha: float = 0.55,
            crop: Optional[Tuple[slice, slice]] = None, contour_only: Sequence[bool] = ()) -> None:
    if crop is not None:
        base = base[crop]
    ax.imshow(base, cmap="gray", interpolation="nearest")
    for i, (m, color) in enumerate(masks):
        mm = m[crop] if crop is not None else m
        if not mm.any():
            continue
        if i < len(contour_only) and contour_only[i]:
            ax.contour(mm.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
        else:
            rgba = np.zeros((*mm.shape, 4))
            rgba[..., :3] = matplotlib.colors.to_rgb(color)
            rgba[..., 3] = mm * alpha
            ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def lr_tags(ax) -> None:
    """Patient-side tags: R index increases to the right of the image."""
    ax.text(0.02, 0.04, "L", transform=ax.transAxes, color="white", fontsize=9, fontweight="bold")
    ax.text(0.94, 0.04, "R", transform=ax.transAxes, color="white", fontsize=9, fontweight="bold")


def square_crop(mask: np.ndarray, margin: int = 12) -> Tuple[slice, slice]:
    """Square crop centred on a 2D mask (keeps every panel the same aspect ratio)."""
    if not mask.any():
        return (slice(None), slice(None))
    r, c = np.where(mask)
    half = int(max(r.max() - r.min(), c.max() - c.min()) / 2) + margin
    rc, cc = int((r.max() + r.min()) / 2), int((c.max() + c.min()) / 2)
    r0 = int(np.clip(rc - half, 0, max(mask.shape[0] - 2 * half, 0)))
    c0 = int(np.clip(cc - half, 0, max(mask.shape[1] - 2 * half, 0)))
    return (slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half))


def bbox_crop(mask: np.ndarray, margin: int = 12) -> Tuple[slice, slice]:
    if not mask.any():
        return (slice(None), slice(None))
    r, c = np.where(mask)
    return (slice(max(r.min() - margin, 0), r.max() + margin + 1),
            slice(max(c.min() - margin, 0), c.max() + margin + 1))
