# PyTorch Out-of-Distribution Detection

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?labelColor=gray"></a>

[comment]: <> (<a href="https://git.kondas.de/kkirchheim/anosuit/-/commits/master"><img alt="pipeline status" src="https://git.kondas.de/kkirchheim/anosuit/badges/master/pipeline.svg"/></a>)
[comment]: <> (<a href="https://git.kondas.de/kkirchheim/anosuit/commits/master"><img alt="Coverage" src="https://git.kondas.de/kkirchheim/anosuit/badges/master/coverage.svg"><a/>)

Python library to accelerate research in fields related to Out-of-Distribution Detection, Open-Set Recognition,
Novelty Detection, Confidence Estimation and Anomaly Detection based on Deep Neural Networks (with PyTorch).

This library provides

- Objective Functions
- OOD Detection Methods
- Datasets used in academic literature
- Neural Network Architectures used in academic literature
- Useful Utilities

Is provided in the hope to speed up research and to facilitate reproducibility.


## Setup

```shell
pip install pytood
```

### Optional Dependencies
For OpenMax, you will have to install `libmr`, which is currently broken.
You will have to install `cython` and `libmr` afterwards by manually.


## License
The code is licensed Apache 2.0. We have taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc.
The legal implications of using pre-trained models in commercial services are, to our knowledge, not fully understood.
