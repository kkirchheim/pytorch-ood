"""
Training script for models hosted in the pytorch-ood model registry.

Trains a classifier on CIFAR-10/100 or ImageNet-200, optionally with an
auxiliary outlier dataset for methods like Outlier Exposure or Energy
Regularization. Configuration is managed with Hydra; see ``configs/``.

Examples:

.. code-block:: bash

    python train.py loss=logitnorm seed=0
    python train.py dataset=cifar100 loss=energy outliers=tinyimages300k
    python train.py dataset=imagenet200 model=resnet-18 \\
        dataset.root=/path/to/imagenet epochs=90 batch_size=256
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import hydra
import numpy as np
import torch
import torchvision.transforms as tvt
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from pytorch_ood.dataset.img import ImageNet200
from pytorch_ood.model import load_model
from pytorch_ood.utils import ToRGB, ToUnknown

log = logging.getLogger(__name__)

CIFAR_DATASETS = {"cifar10": CIFAR10, "cifar100": CIFAR100}

# training methods that require an auxiliary outlier dataset
REQUIRES_OUTLIERS = {"oe", "energy", "entropic", "vos"}


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def instantiate_without(cfg_node: DictConfig, *keys: str, **kwargs: Any) -> Any:
    """
    Instantiate a config node, dropping metadata keys (e.g. ``arch_name``)
    that the target class does not accept.
    """
    node: Dict[str, Any] = {k: v for k, v in cfg_node.items() if k not in keys}
    return instantiate(node, **kwargs)


def infinite(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def build_transforms(cfg: DictConfig):
    normalize = tvt.Normalize(mean=list(cfg.dataset.mean), std=list(cfg.dataset.std))

    if cfg.dataset.get("kind", "cifar") == "imagenet":
        # raw ImageNet includes a handful of grayscale/CMYK images; ToRGB()
        # normalizes channel count before ToTensor/Normalize
        train = tvt.Compose(
            [
                tvt.RandomResizedCrop(cfg.dataset.image_size),
                tvt.RandomHorizontalFlip(),
                ToRGB(),
                tvt.ToTensor(),
                normalize,
            ]
        )
        test = tvt.Compose(
            [
                tvt.Resize(cfg.dataset.pre_size),
                tvt.CenterCrop(cfg.dataset.image_size),
                ToRGB(),
                tvt.ToTensor(),
                normalize,
            ]
        )
        return train, test

    train = tvt.Compose(
        [
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            normalize,
        ]
    )
    test = tvt.Compose([tvt.ToTensor(), normalize])
    return train, test


def build_datasets(cfg: DictConfig, train_transform, test_transform):
    if cfg.dataset.get("kind", "cifar") == "imagenet":
        train_set = ImageNet200(root=cfg.dataset.root, split="train", transform=train_transform)
        # the 1000-image "val" split is OpenOOD's held-out set for model/hyperparameter
        # selection; the 9000-image "test" split is the in-distribution OOD-benchmark set
        # and must stay unseen during training
        eval_set = ImageNet200(root=cfg.dataset.root, split="val", transform=test_transform)
        return train_set, eval_set

    dataset_cls = CIFAR_DATASETS[cfg.dataset.name]
    train_set = dataset_cls(
        root=cfg.dataset.root, train=True, download=True, transform=train_transform
    )
    eval_set = dataset_cls(
        root=cfg.dataset.root, train=False, download=True, transform=test_transform
    )
    return train_set, eval_set


def evaluate(model, loader, device, num_classes) -> float:
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            accuracy.update(logits, y.to(device))
    return accuracy.compute().item()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    method = cfg.loss.method_name

    if method in REQUIRES_OUTLIERS and cfg.outliers.name == "none":
        raise ValueError(
            f"Training method '{method}' requires an auxiliary outlier dataset, "
            f"e.g. outliers=tinyimages300k"
        )

    set_seed(cfg.seed, cfg.deterministic)
    device = cfg.device

    train_transform, test_transform = build_transforms(cfg)
    train_set, test_set = build_datasets(cfg, train_transform, test_transform)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    outlier_iter: Optional[Iterator] = None
    if cfg.outliers.name != "none":
        outlier_set = instantiate_without(
            cfg.outliers, "name", transform=train_transform, target_transform=ToUnknown()
        )
        outlier_loader = DataLoader(
            outlier_set,
            batch_size=cfg.oe_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        outlier_iter = infinite(outlier_loader)
        log.info(f"Using {len(outlier_set)} outliers from {cfg.outliers.name}")

    model = instantiate_without(cfg.model, "arch_name").to(device)
    if cfg.init_from is not None:
        log.info(f"Initializing weights from '{cfg.init_from}'")
        model.load_state_dict(load_model(cfg.init_from).state_dict())

    criterion_kwargs: Dict[str, Any] = {}
    if method == "vos":
        weights_energy = torch.nn.Linear(cfg.dataset.num_classes, 1)
        # per the VOSRegLoss docstring: the energy re-weighting must start
        # non-negative, since it is passed through relu() before use
        torch.nn.init.uniform_(weights_energy.weight)
        criterion_kwargs = {
            "logistic_regression": torch.nn.Linear(1, 2),
            "weights_energy": weights_energy,
        }
    criterion = instantiate_without(cfg.loss, "method_name", **criterion_kwargs).to(device)
    # some losses (e.g. VOSRegLoss) carry their own learnable parameters (the
    # weighted-energy / logistic-regression heads); include them so they actually train
    optimizer = instantiate(
        cfg.optimizer, params=list(model.parameters()) + list(criterion.parameters())
    )

    # OpenOOD's recipe (BaseTrainer, used for every dataset including CIFAR):
    # a LambdaLR cosine schedule stepped once per training iteration (not per
    # epoch) over epochs * len(train_loader) total steps, with no warmup; see
    # scheduler.per_iteration_cosine.
    total_steps = cfg.epochs * len(train_loader)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer, total_steps=total_steps)

    best_accuracy = 0.0
    accuracy = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            if outlier_iter is not None:
                x_out, y_out = next(outlier_iter)
                x = torch.cat([x, x_out])
                y = torch.cat([y, y_out])

            optimizer.zero_grad()
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        accuracy = evaluate(model, test_loader, device, cfg.dataset.num_classes)
        log.info(
            f"Epoch {epoch + 1}: loss={epoch_loss / len(train_loader):.4f} "
            f"accuracy={accuracy:.4f} lr={scheduler.get_last_lr()[0]:.5f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, output_dir / "model.pt")
            if criterion_kwargs:
                criterion_state = {k: v.cpu() for k, v in criterion.state_dict().items()}
                torch.save(criterion_state, output_dir / "criterion.pt")

    metrics = {"final_accuracy": accuracy, "best_accuracy": best_accuracy}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    main()
