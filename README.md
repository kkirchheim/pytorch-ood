# PyTorch Out-of-Distribution Detection

<a href="https://pypi.org/project/pytorch-ood/">
    <img src="https://img.shields.io/pypi/v/pytorch-ood.svg?color=brightgreen"/>
</a>
<a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?logo=python&logoColor=white"/>
</a>
<a href="https://black.readthedocs.io/en/stable/">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?labelColor=gray"/>
</a>
<a href="">
    <img src="https://static.pepy.tech/badge/pytorch-ood"/>
</a>
<a>
    <img src="https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/pipeline.svg"/>
</a>
<a>
    <img src="https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/coverage.svg"/>
</a>
<a href='https://pytorch-ood.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/pytorch-ood/badge/?version=latest' alt='Documentation Status'/>
</a>

Python library to accelerate research in Out-of-Distribution Detection, as well as related
fields such as Open-Set Recognition, Novelty Detection, Confidence Estimation and Anomaly Detection
based on Deep Neural Networks (with PyTorch).

This library provides

- Objective Functions
- OOD Detection Methods
- Datasets used in academic literature
- Neural Network Architectures used in academic literature, as well as pretrained weights
- Useful Utilities

and was created with the aim to speed up research and to facilitate reproducibility.
It is designed such that it should integrate seamlessly with frameworks that enable the scaling of model training,
like [pytorch-lightning](https://www.pytorchlightning.ai/).


## Installation

```shell
pip install pytorch-ood
```

### Required Dependencies

* `torch`
* `torchvision`
* `scipy`
* `torchmetrics`


### Optional Dependencies

* `libmr` for OpenMax Detector
* `pandas` for the Cub200 Dataset

For OpenMax, you will have to install `libmr`, which is currently broken.
You will have to install `cython` and `libmr` afterwards manually.


## Quick Start
Load model pre-trained with energy regularization, and predict on some dataset `data_loader` using
Energy-based outlier scores.

```python
from src.pytorch_ood.model import WideResNet
from src.pytorch_ood import NegativeEnergy
from src.pytorch_ood.utils import OODMetrics

# create Neural Network
model = WideResNet.from_pretrained("er-cifar10-tune").eval().cuda()

# create detector
detector = NegativeEnergy(model)

# evaluate
metrics = OODMetrics()

for x, y in data_loader:
    metrics.update(detector(x.cuda()), y)

print(metrics.compute())
```


## Implemented Detectors

| Detector       | Reference     |
|--------------|-----------------|
| OpenMax      | [[1]](#bendale2016towards)  |
| ODIN         |   [[2]](#liang2018enhancing)   |
| Mahalanobis      |  [[3]]()  |
| Monte Carlo Dropout      |  [[4]]() |
| Softmax Thresholding Baseline | [[5]]() |
| Energy-Based OOD Detection | [[6]](#liu2020energy) |

## Implemented Objective Functions

| Objective Function       | Reference     |
|--------------|---------------------------|
| Objectosphere      | [[7]]() |
| Outlier Exposure   | [[8]]()  |
| Deep SVDD          | [[9]]()  |
| II Loss           | [[10]]()  |
| CAC Loss           | [[11]]()  |
| Energy Regularization | [[6]](#liu2020energy)  |
| Center Loss           | [[12]]()  |

## Cite pytorch-ood
If you use this package in your research, please consider citing it.
To appear in
```text
@article{kirchheim2022pytorch,
	author = {Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
	journal = {CVPR Workshop for Human-centered Intelligent Services: Safety and Trustworthy},
	number = {},
	pages = {},
	publisher = {IEEE},
	title = {PyTorch-OOD: A Library for Out-of-Distribution Detection based on PyTorch},
	year = {2022}
}
```

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


## References

<a name="bendale2016towards">[1] OpenMax (2016)</a> *Towards open set deep networks*, CVPR

<a name="liang2018enhancing">[2] ODIN (2018) </a> *Enhancing the reliability of out-of-distribution image detection in neural networks*, ICLR

<a name="lee2018simple">[3] Mahalanobis (2018) </a> *A simple unified framework for detecting out-of-distribution samples and adversarial attacks*, NEURIPS

<a name="">[4] ... </a>

<a name="">[5] ... </a>

<a name="liu2020energy">[6] Energy-Based OOD (2020)</a> *Energy-based Out-of-distribution Detection*, NEURIPS

<a name="">[7] ... </a>

<a name="">[8] ... </a>

<a name="">[9] ... </a>

<a name="">[10] ... </a>

<a name="">[11] ... </a>

<a name="">[12] ... </a>
