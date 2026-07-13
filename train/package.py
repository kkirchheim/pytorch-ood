"""
Package a finished training run for upload to the HuggingFace model repository.

Takes a Hydra run directory (containing ``model.pt``, ``metrics.json`` and
``.hydra/config.yaml``) and assembles the upload layout::

    <output-dir>/<arch>/<dataset>/<loss>/<seed>/
    ├── config.yaml        # fully resolved training configuration
    ├── metrics.json
    └── model-<sha256[:16]>.pt

Uploading and registering the model are deliberately manual steps: the script
prints the upload command and a ``ModelEntry`` snippet for
``src/pytorch_ood/model/registry.py``.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

HF_REPO = "kkirchheim/pytorch-ood-models"

ENTRY_TEMPLATE = """\
ModelEntry(
    key="{key}",
    arch="{arch}",
    dataset="{dataset}",
    loss="{loss}",
    seed="{seed}",
    arch_class="{arch_class}",
    arch_kwargs={arch_kwargs},
    url="https://huggingface.co/{repo}/resolve/main/{key}/{ckpt_name}",
    sha256="{sha256}",
    preprocessing=ImagePreprocessing(mean={mean}, std={std}),
    metrics={metrics},
    description="",
)"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Hydra run directory")
    parser.add_argument("--output-dir", type=Path, default=Path("zoo"))
    parser.add_argument("--checkpoint", default="model.pt")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra" / "config.yaml")
    OmegaConf.resolve(cfg)

    arch = cfg.model.arch_name
    dataset = cfg.dataset.name
    loss = cfg.loss.method_name
    seed = f"s{cfg.seed}"
    key = f"{arch}/{dataset}/{loss}/{seed}"

    checkpoint = args.run_dir / args.checkpoint
    digest = sha256_of(checkpoint)
    ckpt_name = f"model-{digest[:16]}.pt"

    metrics = json.loads((args.run_dir / "metrics.json").read_text())

    target = args.output_dir / key
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, target / ckpt_name)
    shutil.copy(args.run_dir / "metrics.json", target / "metrics.json")
    OmegaConf.save(cfg, target / "config.yaml")

    entry = ENTRY_TEMPLATE.format(
        key=key,
        arch=arch,
        dataset=dataset,
        loss=loss,
        seed=seed,
        arch_class=cfg.model._target_,
        arch_kwargs={k: v for k, v in cfg.model.items() if k not in ("_target_", "arch_name")},
        repo=HF_REPO,
        ckpt_name=ckpt_name,
        sha256=digest,
        mean=tuple(cfg.dataset.mean),
        std=tuple(cfg.dataset.std),
        metrics=dict(metrics),
    )

    print(f"Packaged '{key}' into {target}\n")
    print("Upload with:\n")
    print(f"    huggingface-cli upload {HF_REPO} {args.output_dir / arch} {arch}\n")
    print("Registry entry for src/pytorch_ood/model/registry.py:\n")
    print(entry)


if __name__ == "__main__":
    main()
