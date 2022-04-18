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

## Implemented Methods

| Method       | Reference     |
|--------------|-----------|
| OpenMax      |   |
| ODIN |      |
| Mahalanobis      |   |
| Monte Carlo Dropout      |   |
| Softmax Thresholding Baseline      |   |
| Energy Based OOD Detection      |   |
| Objkectosphere      |   |
| Outlier Exposure      |   |
| Deeo SVDD      |   |

## Installation

```shell
pip install pytorch-ood
```

### Optional Dependencies
For OpenMax, you will have to install `libmr`, which is currently broken.
You will have to install `cython` and `libmr` afterwards by manually.



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
