# Training

Reproducible training scripts for the models hosted in the
[pytorch-ood model registry](https://huggingface.co/kkirchheim/pytorch-ood-models).
Configuration is managed with [Hydra](https://hydra.cc); models are identified as
`arch/dataset/loss/seed` (e.g. `wrn-40-2/cifar10/logitnorm/s0`).

## Setup

```bash
pip install -e ..            # pytorch-ood itself
pip install -r requirements.txt
```

## Training

```bash
# wrn-40-2/cifar10/logitnorm/s0
python train.py loss=logitnorm seed=0

# wrn-40-2/cifar100/oe/s0 (requires the auxiliary outlier dataset)
python train.py dataset=cifar100 loss=oe outliers=tinyimages300k seed=0

# fine-tune from a registry model
python train.py loss=energy outliers=tinyimages300k init_from=wrn-40-2/cifar10/crossentropy

# resnet-18/imagenet200/crossentropy/s0 (requires a torchvision-compatible
# ImageNet directory with train/, val/ and meta.bin; see pytorch_ood.dataset.img.ImageNet200)
python train.py dataset=imagenet200 model=resnet-18 \
    dataset.root=/path/to/imagenet epochs=90 batch_size=256 seed=0

# resnet-18/imagenet200/entropic/s0 (outlier-exposure-style training on ImageNet-200; the
# auxiliary dataset is ImageNet-800, the 800 ImageNet-1K classes disjoint from ImageNet-200 -
# see pytorch_ood.dataset.img.ImageNet800. Same root works for both, since it's a view on the
# same torchvision-compatible ImageNet directory)
python train.py dataset=imagenet200 model=resnet-18 loss=entropic outliers=imagenet800 \
    dataset.root=/path/to/imagenet outliers.root=/path/to/imagenet epochs=90 batch_size=256 seed=0
```

Each run writes `model.pt` (best checkpoint by test/val accuracy), `metrics.json`, and the
resolved configuration (`.hydra/config.yaml`) into the Hydra output directory
(`outputs/<date>/<time>/`). For `imagenet200`, model selection during training uses the
1000-image OpenOOD "val" split; the 9000-image "test" split is reserved for the OOD
benchmark and is never seen during training.

Training runs on a single GPU. On machines without internet access (e.g. SLURM compute
nodes), pre-populate the dataset directory and the `torch.hub` checkpoint cache
(configurable via `TORCH_HOME`) beforehand.

## Packaging for upload

`package.py` converts a finished run into the directory layout of the HuggingFace
repository and prints the upload command plus a ready-to-paste registry entry:

```bash
python package.py outputs/2026-07-13/12-00-00 --output-dir zoo
```

Uploading and registering the model are deliberately manual steps: review the metrics,
upload with `huggingface-cli upload`, and add the printed `ModelEntry` to
`src/pytorch_ood/model/registry.py`.
