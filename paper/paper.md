---
title: "pytorch-ood: A Unified PyTorch Library for Out-of-Distribution Detection"
tags:
  - Python
  - PyTorch
  - out-of-distribution detection
  - anomaly detection
  - machine learning safety
authors:
  - name: Konstantin Kirchheim
    orcid: 0000-0001-5819-7692
    affiliation: "1"
    corresponding: true
  - name: Theo Langer
    orcid: 0009-0001-9393-4747
    affiliation: "1"
  - name: Frank Ortmeier
    orcid: 0000-0001-6186-4142
    affiliation: "1"
affiliations:
  - name: Otto-von-Guericke University Magdeburg, Germany
    index: 1
date: 14 April 2026
bibliography: paper.bib
---

# Summary

Out-of-distribution (OOD) detection is a central problem in modern machine learning, especially in safety-critical settings where predictive systems must recognize when an input when an input no longer follows the training data distribution @yang2021generalized. Although the field has developed rapidly, practical experimentation remains fragmented.
Implementations are often tied to individual papers, method interfaces vary considerably, and evaluation protocols can be difficult to reproduce.
`pytorch-ood` aims to address this problem by providing a unified, research-oriented software library for OOD detection in the PyTorch ecosystem @paszke2019pytorch.
The package combines a broad collection of detectors, training objectives, datasets, pretrained models, and evaluation utilities behind a consistent interface, thereby reducing engineering overhead and making controlled comparisons easier to conduct.

The library is designed for researchers who need both breadth and modularity. Rather than focusing on a single benchmark or a narrow family of methods, `pytorch-ood` aims to provide a reusable framework for OOD detection research.
The library is accompanied by extensive documentation and unit tests, supporting reliable reuse and facilitating reproducible experimentation.

# Statement of need

Research on OOD detection faces a recurring methodological problem: many published methods are conceptually comparable but difficult to compare in practice. Small implementation differences in preprocessing, score orientation, model wrapping, or metric computation can materially affect conclusions. At the same time, reproducing baselines often requires re-implementing substantial amounts of auxiliary code for dataset handling, evaluation, and model integration. This duplication slows progress and makes empirical claims harder to audit.

`pytorch-ood` was developed to reduce this friction. The package provides a unified API for detectors and related components while covering a broad range of methods and benchmark datasets. This enables reuse of trained models across detectors, evaluation under a shared interface, and integration of new ideas without rebuilding surrounding infrastructure.
The resulting workflow is lighter than bespoke per-paper implementations and more flexible than fixed benchmark pipelines, establishing `pytorch-ood` as a middle layer between individual research code and benchmark frameworks.


### Comparison to existing tools

Several tools for OOD detection exist (many released after `pytorch-ood`). Some, such as OpenOOD @zhang2023openood, emphasize standardized benchmark pipelines, while others like FrOODo @stieber2022froodo focus on specific domains such as medical applications.

`pytorch-ood` occupies a different point in the design space.
Rather than prescribing a fixed evaluation regime or domain, it provides a flexible, PyTorch-native layer for composing detectors, models, datasets, and benchmarks under a unified interface.
This makes it well suited for exploratory research, baseline construction, and rapid integration of new ideas. Accordingly, it is best understood as complementary to benchmark-centric frameworks, prioritizing modularity and reuse.


# Software Design


The core architectural principle of `pytorch-ood` is interface unification.
All OOD detection methods implement a shared `Detector` abstraction with a `predict()` method for mapping inputs to outlier scores and, where needed, an optional `fit()` method for calibration or training.

Beyond this common interface, the library provides specialized detector types such as `LogitsDetector` and `FeaturesDetector`, which differ by the representation they consume while preserving the base `Detector` interface.
These subclasses additionally expose representation-specific methods (e.g., `predict_logits()`, `predict_features()`), enabling detectors to operate directly on precomputed intermediate representations.
For instance, logit-based detectors operate directly on model outputs, while feature-based detectors consume intermediate representations.

```python
# reuse logits across detectors
logits = model(x)
scores1 = detector1.predict_logits(logits)
scores2 = detector2.predict_logits(logits)
```

This separation makes detectors easy to interchange in experiments and enables computational reuse, since extracted logits or features can be shared across multiple methods.

![architecture](arch.png)


### Metrics and Evaluation Conventions

OOD evaluation is often affected by inconsistent conventions, such as differing score orientations or label encodings, which can lead to incomparable results.

To ensure consistency, `pytorch-ood` enforces a canonical evaluation protocol:

1. **Problem formulation**: OOD detection is treated as binary classification between in-distribution (ID) and out-of-distribution (OOD) samples.

2. **Score semantics**: Detectors produce scores $s(x)$ where higher values indicate a higher likelihood of being OOD.

3. **Label encoding**: OOD samples have labels $y < 0$, while ID samples retain standard class labels $y \geq 0$.

These conventions are applied uniformly across detectors, datasets, and benchmarks, enabling interchangeable components and directly comparable results without additional adjustments.

### Benchmarks

Additionally, we define a benchmark interface that exposes an `evaluate()` method, which takes a detector.
This interface enables straightforward reproduction of benchmark protocols from prior work while still allowing custom evaluation pipelines when needed.

For example, a minimal evaluation workflow to replicate the OpenOOD v1.5 CIFAR10  benchmark for the pre-trained model from one of the first OOD Detection benchmark papers by Hendrycks *et al.* @hendrycks2016baseline can be written directly as:

```python
from pytorch_ood.detector import EnergyBased, MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.benchmark import CIFAR10_OpenOOD

device = "cuda"

model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval()
preprocess = WideResNet.transform_for("cifar10-pt")
ebo = EnergyBased(model)
msp = MaxSoftmax(model)

benchmark = CIFAR10_OpenOOD(root="data", transform=preprocess, cache=True)

metrics = benchmark.evaluate(
  [ebo, msp],
  loader_kwargs={"batch_size": 64},
  device=device
)

print(metrics)
```

Since both detectors used in this example implement the LogitsDetector interface, these logits will be extracted from the model only once.


# Research Impact Statement

The `pytorch-ood` library has seen sustained adoption within the research community. The associated publication has been cited more than 60 times, and the repository has accumulated over 300 GitHub stars with contributions from 9 developers.
Overall, `pytorch-ood` serves both as reusable research infrastructure and as a foundation for further work in OOD detection and machine learning safety.



# AI Usage Disclosure
Development of `pytorch-ood` began prior to the widespread availability of modern generative AI systems.
More recent contributions, including parts of the codebase and documentation, were developed with the assistance of generative AI tools.
All such contributions were reviewed, tested, and validated by the authors.
Generative AI was also used for editorial support in the preparation of this paper.
