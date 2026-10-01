"""Render the sample images shown on dataset pages into docs/_static/datasets/.

Only for datasets whose license allows showing samples in the docs; each figure's
caption in the dataset docstring credits the source and names the license. Needs
the datasets, which it downloads into ``root`` on first use:

    python docs/tools/dataset_samples.py --root data
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pytorch_ood.dataset.img import CIFAR100GAN, MNISTC, SuMNIST

OUT = Path(__file__).resolve().parent.parent / "_static" / "datasets"
LABEL = "#6b7280"  # readable on the light and the dark theme
OOD = "#e8871e"


def save(fig, name: str, dpi: int = 100, lossless: bool = False) -> None:
    """Transparent WebP, so one file serves the light and the dark theme. Lossless
    suits line art and digits, lossy the photo-like sample grids."""
    path = OUT / f"{name}.webp"
    options = {"lossless": True} if lossless else {"quality": 85, "alpha_quality": 100}
    fig.savefig(path, transparent=True, dpi=dpi, pil_kwargs=options)
    plt.close(fig)
    print("wrote", path)


def mnistc(root: str) -> None:
    """The same test digit under each of the 16 corruptions."""
    index = 7
    fig, axes = plt.subplots(2, 8, figsize=(8, 2.4))
    for ax, subset in zip(axes.flat, MNISTC.subsets):
        img, _ = MNISTC(root, subset=subset, split="test", download=True)[index]
        ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(subset.replace("_", " "), fontsize=8, color=LABEL, pad=3)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.08, hspace=0.3)
    save(fig, "mnistc", dpi=200, lossless=True)  # sharp labels on high-density screens


def cifar100gan(root: str) -> None:
    """A 4 x 12 grid of samples."""
    data = CIFAR100GAN(root, download=True)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 12, figsize=(8, 2.75))
    for ax, i in zip(axes.flat, rng.choice(len(data), size=48, replace=False)):
        img, _ = data[int(i)]
        ax.imshow(np.asarray(img))
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
    save(fig, "cifar100gan")


def sumnist(root: str) -> None:
    """Test images with their digit boxes: four normal ones, four anomalies."""
    data = SuMNIST(root, train=False, download=True)
    # the dataset's own "anomaly" field is unreliable, so derive it from the digits
    sums = data.y.sum(dim=1)
    normal = (sums == 20).nonzero().flatten()[:4].tolist()
    anomalous = (sums != 20).nonzero().flatten()[::300][:4].tolist()
    fig, axes = plt.subplots(1, 8, figsize=(8, 1.45))
    for ax, i in zip(axes, normal + anomalous):
        img, target = data[i]
        total = int(target["labels"].sum())
        color = LABEL if total == 20 else OOD
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for x0, y0, x1, y1 in target["boxes"].tolist():
            ax.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, lw=0.8)
            )
        ax.set_title(f"sum = {total}", fontsize=8, color=color, pad=3)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.08)
    save(fig, "sumnist", dpi=200, lossless=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data", help="dataset directory")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mnistc(args.root)
    cifar100gan(args.root)
    sumnist(args.root)
