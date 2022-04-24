# PyTorch Out-of-Distribution Detection

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?labelColor=gray"></a>

Python library to accelerate research in fields related to Out-of-Distribution Detection, Open-Set Recognition,
Novelty Detection, Confidence Estimation and Anomaly Detection based on Deep Neural Networks (with PyTorch).

This library implements

- Objective Functions
- OOD Detection Methods
- Datasets used in academic literature
- Neural Network Architectures used in academic literature, as well as pretrained weights
- Useful Utilities

It is provided with the aim to speed up research and to facilitate reproducibility.

## Installation

```shell
pip install pytorch-ood
```

### Optional Dependencies
For OpenMax, you will have to install `libmr`, which is currently broken.
You will have to install `cython` and `libmr` afterwards manually.


## Quick Start
Load model pre-trained with energy regularization, and predict on some dataset `data_loader` using
Energy-based outlier scores.
```python
from pytorch_ood.model import WideResNet
from pytorch_ood import NegativeEnergy
from pytorch_ood.metrics import OODMetrics

model = WideResNet.from_pretrained("er-cifar10-tune").eval().cuda()
detector = NegativeEnergy(model)

metrics = OODMetrics()

for x, y in data_loader:
    metrics.update(detector(x.cuda()), y)

print(metrics.compute())
```


## Implemented Methods

| Method       | Reference     |
|--------------|-----------|
| OpenMax      |   |
| ODIN |      |
| Mahalanobis      |   |
| Monte Carlo Dropout      |   |
| Softmax Thresholding Baseline      |   |
| Energy Based OOD Detection      |   |
| Objectosphere      |   |
| Outlier Exposure      |   |
| Deep SVDD      |   |


## Roadmap
- [ ] add additional OOD methods
- [ ] add more datasets, e.g. for audio and video
- [ ] implement additional tests
- [ ] migrate to [DataPipes](https://github.com/pytorch/data)

## Contributing
We encourage everyone to contribute to this project by adding implementations of OOD Detection methods, datasets etc,
or check the existing implementations for bugs.

## License
The code is licensed under Apache 2.0. We have taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc.
The legal implications of using pre-trained models in commercial services are, to our knowledge, not fully understood.
