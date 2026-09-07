---
title: "pytorch-ood: A Unified PyTorch Library for Out-of-Distribution Detection"
tags:
  - Python
  - PyTorch
  - Out-of-Distribution Detection
  - Anomaly Detection
  - Machine Learning Safety
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
date: 15 April 2026
bibliography: paper.bib
---

# Summary

Machine learning models, particularly deep neural networks can produce unreliable predictions when they encounter data that differs from what they were trained on. Detecting such cases - commonly referred to as out-of-distribution (OOD) detection - is particularly important in safety-critical applications [@yang2021generalized].
Despite rapid progress in the field, practical experimentation remains fragmented. Implementations are often tied to individual papers, interfaces differ across methods, and evaluation setups can be difficult to reproduce.

`pytorch-ood` is a Python library that provides a unified framework for OOD detection in the PyTorch deep learning ecosystem [@paszke2019pytorch]. It encompasses a broad range of detection methods, training objectives, datasets, pre-trained models, and evaluation tools under a consistent interface. This allows users to apply different methods, compare them under shared conditions, and integrate them into existing workflows with minimal additional code.
The library is designed to support flexible experimentation rather than a single fixed benchmark. It is accompanied by extensive documentation and unit tests, enabling reliable reuse and facilitating reproducible research.


# Statement of need

Research on OOD detection faces a recurring methodological problem: many published methods are conceptually comparable but difficult to compare in practice. Small implementation differences in preprocessing, score orientation, or metric computation can materially affect conclusions. At the same time, reproducing baselines often requires re-implementing substantial amounts of auxiliary code for dataset handling, evaluation, and model integration. This duplication slows progress and makes empirical results harder to audit.

`pytorch-ood` was developed to reduce this friction. The package provides a unified API for detectors and related components while covering a broad range of methods and benchmark datasets. This enables the reuse of trained models across detectors, evaluation under a shared interface, and the integration of new methods without rebuilding surrounding infrastructure.

The library is intended for researchers developing and evaluating OOD detection methods in PyTorch. It reduces engineering overhead compared to bespoke per-paper implementations while remaining more flexible than fixed benchmark pipelines. In this sense, `pytorch-ood` provides a unified and reproducible interface for OOD detection experiments, bridging the gap between ad-hoc research code and standardized benchmarking frameworks.

An earlier version of the library was introduced in [@kirchheim2022pytorch]. The present work reflects a substantially extended and matured system.


# State of the field

Several tools for OOD detection exist, many of which were released after `pytorch-ood`. Frameworks such as OpenOOD [@yang2022openood] emphasize standardized benchmark pipelines, enabling consistent large-scale evaluation across datasets and methods. Other tools, such as FrOoDo [@stieber2022froodo], focus on specific application domains, for example medical imaging.

`pytorch-ood` targets a complementary use case. Rather than prescribing a fixed evaluation pipeline or domain, it provides a flexible, PyTorch-native layer for composing detectors, models, datasets, and benchmarks under a unified interface. This design is motivated by the observation that exploratory research often requires fine-grained control over experimental details, access to intermediate neural representations, and varying training objectives, which can be difficult to achieve within fixed benchmark frameworks.

As a result, `pytorch-ood` is particularly well-suited for baseline construction, ablation studies, and the rapid integration of new methods.
It is therefore best understood as complementary to benchmark-centric frameworks, prioritizing modularity, reuse, and controlled experimentation.


# Software Design

The design of `pytorch-ood` is guided by the goal of enabling *comparable and reusable OOD detection experiments* while maintaining flexibility for research.
A central design choice is interface unification.

### Detectors

All OOD detection methods implement a shared `Detector` abstraction with a `predict()` method that maps inputs to outlier scores and, where required, an optional `fit()` method for calibration or training.
This defines a minimal contract: given an input tensor $x$, a detector produces a scalar outlier score $s(x) \in \mathbb{R}$ indicating how likely it is to be out-of-distribution.
As a result, different detectors can be applied interchangeably and compared directly, without requiring knowledge of their internal implementation or score computation.
The trade-off is a slight restriction on method-specific interfaces in exchange for consistent and comparable usage across detectors.


### Representation-specific Interfaces

OOD detectors can be differentiated based on the kind of input that they work on.
Usually, they operate on neural representations of the input in some layer of a deep neural network.
To further support reuse, the library distinguishes detectors by the type of representation they consume.
In addition to the base interface, specialized classes such as `LogitsDetector` and `FeaturesDetector` operate on logits or intermediate feature representations, respectively.
This design reflects a second trade-off: for different detectors that operate on the same type of representation, the required representations only need to be computed once rather than once per detector:
```python
# reuse logits across detectors
logits = model(x)
scores1 = detector1.predict_logits(logits)
scores2 = detector2.predict_logits(logits)
```

While this separation reduces redundant computation and simplifies controlled comparisons, as multiple methods can be evaluated under identical conditions, it comes at the cost of slightly increased API complexity.


![Overview of Relevant Abstractions](arch.png)


### Evaluation Semantics

A third design principle is the standardization of evaluation semantics.
In practice, OOD detection experiments often differ in label encoding or metric assumptions, which can lead to subtle and hard-to-detect errors in evaluation.

To mitigate this, `pytorch-ood` enforces a canonical evaluation protocol across all components. Datasets follow a unified label convention with OOD samples assigned $y < 0$, and metrics operate under consistent assumptions (such as the score orientation) across detectors and benchmarks.
While this slightly reduces flexibility, it eliminates ambiguity in metric computation and ensures that results are directly comparable across methods.


### Benchmarks

Finally, the library provides a benchmark abstraction that exposes a common `evaluate()` interface.
This enables the reproduction of established evaluation protocols while remaining compatible with custom training setups.
The design balances two competing goals: supporting standardized experiments and preserving the ability to compose new ones.
In practice, benchmark objects allow users to evaluate multiple detectors on shared datasets with minimal boilerplate while retaining full control over other experimental details.

For example, a minimal evaluation workflow to replicate the OpenOOD v1.5 CIFAR-10 benchmark [@yang2022openood] with the pre-trained model and baseline detector from one of the first OOD detection benchmark papers [@hendrycks2016baseline] and an additional, more recent detector [@liu2023gen] can be written directly as:

```python
from pytorch_ood.detector import MaxSoftmax, GEN
from pytorch_ood.model import WideResNet
from pytorch_ood.benchmark import CIFAR10_OpenOOD

model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval()
preprocess = WideResNet.transform_for("cifar10-pt")

msp = MaxSoftmax(model)
gen = GEN(model, gamma=0.1)

benchmark = CIFAR10_OpenOOD(root="data", transform=preprocess)

metrics = benchmark.evaluate(
  [gen, msp],
  loader_kwargs={"batch_size": 64},
  device="cuda",
  cache=True
)

print(metrics)
```
Required datasets and weights will be downloaded automatically.
Since both detectors used in this example implement the `LogitsDetector` interface, logits will be extracted from the deep neural network only once and then passed through both detectors.


# Research Impact Statement

The `pytorch-ood` library has seen sustained adoption within the research community. The associated publication [@kirchheim2022pytorch] has been cited more than 60 times, and the repository has accumulated over 300 GitHub stars with contributions from 9 developers and a substantial number of downloads.
Overall, `pytorch-ood` serves both as reusable research infrastructure and as a foundation for further work in OOD detection and machine learning safety.


# AI Usage Disclosure

Generative AI tools were used during the development of `pytorch-ood` and the preparation of this manuscript.
In particular, we used systems from OpenAI (including ChatGPT and Codex) and Anthropic (Claude) for parts of the codebase, documentation, and manuscript text.
The exact historical model versions used for this assistance cannot be recovered.
These tools provided assistance with code generation and refactoring, drafting and improving documentation, generating testing scaffolding, and editorial support.
All AI-assisted contributions were carefully checked and integrated by the authors. All architectural and design decisions were made by the authors, who take full responsibility for the software and this manuscript.


# Acknowledgements

The authors acknowledge institutional support from Otto-von-Guericke University Magdeburg, Germany. No dedicated financial support was received specifically for the development of `pytorch-ood` or for this JOSS submission.

# References
