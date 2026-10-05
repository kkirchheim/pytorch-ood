"""Draw the example gallery thumbnails into docs/_static/thumbs/, one per section.

The examples are not executed when the docs are built, so sphinx-gallery has no
figures to make thumbnails from. Each example instead names its section's image
with a ``# sphinx_gallery_thumbnail_path`` comment. The images are transparent,
in mid-tone colors that read on the light and the dark theme: blue for
in-distribution, the logo's orange for OOD.

    python docs/tools/gallery_thumbnails.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "_static" / "thumbs"

ID, ID2, OOD = "#3d7fe0", "#7aa7ea", "#e8871e"
LINE, MUTED = "#3d7fe0", "#8b949e"

W, H = 4.0, 2.8  # sphinx-gallery's default thumbnail size, 400 x 280 px at 100 dpi
rng = np.random.default_rng(0)


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.1, 0.12, 0.8, 0.76])
    ax.set_facecolor("none")
    ax.set_axis_off()
    return fig, ax


def save(fig, name: str) -> None:
    path = OUT / f"{name}.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("wrote", path)


def detectors():
    """Score distributions of ID and OOD data, separated by a threshold."""
    fig, ax = canvas()
    x = np.linspace(-4, 6, 400)
    score_id = np.exp(-0.5 * (x / 0.9) ** 2)
    score_ood = np.exp(-0.5 * ((x - 2.8) / 1.1) ** 2) * 0.8
    for y, color in ((score_id, ID), (score_ood, OOD)):
        ax.fill_between(x, y, color=color, alpha=0.35, lw=0)
        ax.plot(x, y, color=color, lw=2.5)
    ax.axvline(1.45, color=LINE, lw=1.5, ls=(0, (4, 3)))
    ax.plot([-4, 6], [0, 0], color=MUTED, lw=1.5)
    save(fig, "detectors")


def loss():
    """Compact class clusters, with outliers pushed away from them."""
    fig, ax = canvas()
    for cx, cy, color in [(-1.6, 0.9, ID), (1.6, 0.9, ID2), (0, -1.5, ID)]:
        points = rng.normal([cx, cy], 0.32, size=(40, 2))
        ax.scatter(*points.T, s=14, color=color, alpha=0.9, lw=0)
    angle = rng.uniform(0, 2 * np.pi, 26)
    radius = rng.uniform(2.6, 3.1, 26)
    ax.scatter(radius * np.cos(angle), radius * np.sin(angle) * 0.75, s=16, color=OOD, lw=0)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.6, 2.6)
    save(fig, "loss")


def segmentation():
    """A per-pixel outlier score map with one anomalous region."""
    fig, ax = canvas()
    ax.set_position([0.08, 0.1, 0.84, 0.8])
    yy, xx = np.mgrid[0:56, 0:80]
    field = 0.25 + 0.12 * np.sin(xx / 7.0) * np.cos(yy / 9.0) + 0.05 * rng.normal(size=xx.shape)
    blob = np.exp(-(((xx - 52) / 7.0) ** 2 + ((yy - 34) / 5.0) ** 2))
    cmap = LinearSegmentedColormap.from_list("score", ["#16325c", ID2, OOD])
    ax.imshow(
        np.clip(field + 0.9 * blob, 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="bicubic"
    )
    save(fig, "segmentation")


def text():
    """Lines of tokens, one of them out of distribution."""
    fig, ax = canvas()
    y = 0.0
    for line in range(6):
        x = 0.0
        for _ in range(rng.integers(3, 6)):
            width = rng.uniform(0.6, 1.8)
            if x + width > 8:
                break
            color = OOD if line == 3 and 2 < x < 5 else (ID if line % 2 else ID2)
            box = FancyBboxPatch(
                (x, y), width, 0.42, boxstyle="round,pad=0,rounding_size=0.18", color=color, lw=0
            )
            ax.add_patch(box)
            x += width + 0.25
        y -= 0.75
    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(-4.3, 0.8)
    save(fig, "text")


def osr():
    """Known classes and unknown samples outside all of them."""
    fig, ax = canvas()
    for i, (cx, cy) in enumerate([(-1.7, 0.8), (1.7, 0.8), (0, -1.3)]):
        color = ID if i % 2 == 0 else ID2
        ax.add_patch(plt.Circle((cx, cy), 1.0, color=color, alpha=0.18, lw=0))
        ax.add_patch(plt.Circle((cx, cy), 1.0, fill=False, color=LINE, lw=1.2, ls=(0, (3, 3))))
        ax.scatter(*rng.normal([cx, cy], 0.3, size=(22, 2)).T, s=12, color=color, lw=0)
    unknown = np.array([(2.9, -1.3), (-2.8, -1.4), (0.1, 2.0), (3.0, 1.9), (-0.4, 0.3)])
    ax.scatter(*unknown.T, s=60, marker="X", color=OOD, lw=0)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    save(fig, "osr")


def metrics():
    """ROC curves."""
    fig, ax = canvas()
    ax.set_aspect("equal")
    fpr = np.linspace(0, 1, 200)
    tpr = 1 - (1 - fpr) ** 6
    ax.fill_between(fpr, tpr, color=ID, alpha=0.22, lw=0)
    ax.plot(fpr, tpr, color=ID, lw=2.8)
    ax.plot(fpr, 1 - (1 - fpr) ** 2, color=ID2, lw=2)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=MUTED, lw=1.5)
    ax.scatter([0.05], [1 - 0.95**6], s=40, color=OOD, zorder=3)
    save(fig, "metrics")


def hpo():
    """A grid search, the best configuration marked."""
    fig, ax = canvas()
    ax.set_position([0.2, 0.1, 0.6, 0.8])
    grid = np.add.outer(-((np.arange(5) - 3.2) ** 2), -((np.arange(7) - 2.1) ** 2) / 2.0)
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    # separate squares, shaded by opacity, so no background color is needed
    for i in range(5):
        for j in range(7):
            cell = plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, color=ID, lw=0)
            cell.set_alpha(0.2 + 0.8 * grid[i, j])
            ax.add_patch(cell)
    best_i, best_j = np.unravel_index(grid.argmax(), grid.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, color=OOD, lw=3))
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(4.6, -0.6)
    ax.set_aspect("equal")
    save(fig, "hpo")


def benchmarks():
    """A leaderboard."""
    fig, ax = canvas()
    scores = np.array([0.93, 0.88, 0.86, 0.81, 0.77, 0.72])
    rows = np.arange(len(scores))[::-1]
    ax.barh(rows, np.ones_like(scores), height=0.62, color=MUTED, alpha=0.35)
    ax.barh(rows, scores, height=0.62, color=[OOD] + [ID] * (len(scores) - 1))
    ax.set_xlim(0, 1.02)
    save(fig, "benchmarks")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (detectors, loss, segmentation, text, osr, metrics, hpo, benchmarks):
        draw()
