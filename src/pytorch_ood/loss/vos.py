"""
Parts of this code are taken from
 https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..api import LossInfo, Paper, Representation, Task
from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction, drop_unknown, is_known, is_unknown


class VOSRegLoss(nn.Module):
    """
    Implements the loss function from  *VOS: Learning what you don’t know by virtual outlier synthesis*
    without the synthesising of virtual outliers.
    The loss adds a regularization term to the cross-entropy that aims to increase the (weighted) energy gap between
    ID and OOD samples.

    The regularization term is defined as:

    .. math::
        \\mathcal{L}_{reg} = \\mathbb{E}_{v \\sim V} \\left[ -\\text {log}\\frac{1}{1+\\exp(-\\phi(E(v)))}
        \\right] +  \\mathbb{E}_{x \\sim D} \\left[ -\\text {log} \\frac{\\exp(-\\phi(E(x)))}{1+
        \\exp(-\\phi(E(x)))}\\right]


    where :math:`\\phi` is a possibly non-linear function, :math:`E` is the weighted energy
    and :math:`V` and :math:`D` are the distributions of the (possibly virtual) outliers and the ID data respectively.
    The weighted energy is :math:`E(x) = -\\log \\sum_i w_i e^{f_i(x)}` with
    :math:`w_i = \\mathrm{ReLU}(\\text{weights\\_energy.weight}_i)`, and the total loss is
    :math:`\\mathcal{L}_{CE} + \\alpha \\mathcal{L}_{reg}`.

    For initialisation of :math:`\\phi` and the weights for weighted energy:

    .. code :: python

        phi = torch.nn.Linear(1, 2)
        weights = torch.nn.Linear(num_classes, 1)
        torch.nn.init.uniform_(weights.weight)
        criterion = VOSRegLoss(phi, weights)

    .. note ::
        This implementation does not generate synthetic outliers. For this feature, see  :class:`pytorch_ood.loss.vos.VirtualOutlierSynthesizingRegLoss`.
    """

    info = LossInfo(
        paper=Paper(
            title="VOS: Learning What You Don't Know by Virtual Outlier Synthesis",
            venue="ICLR",
            year=2022,
            url="https://arxiv.org/pdf/2202.01197.pdf",
            code="https://github.com/deeplearning-wisc/vos/",
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
        inputs={Representation.LOGITS},
        supervised=True,
    )

    def __init__(
        self,
        logistic_regression: torch.nn.Linear,
        weights_energy: torch.nn.Linear,
        alpha: float = 0.1,
        device: str = "cpu",
        reduction: str = "mean",
    ):
        """
        :param logistic_regression: :math:`\\phi` function, mapping energies of shape :math:`N \\times 1` to
            :math:`N \\times 2` logits (ID and OOD). Can be for example a linear layer.
        :param weights_energy: ``torch.nn.Linear(num_classes, 1)`` whose (ReLU-ed) weight rescales the
            exponentiated logits in the weighted energy
        :param alpha: weighting parameter :math:`\\alpha`.
        :param reduction: reduction method to apply, one of ``mean``, ``sum`` or ``none``
        :param device: For example ``cpu`` or ``cuda:0``
        """
        super(VOSRegLoss, self).__init__()
        self.logistic_regression = logistic_regression
        self.weights_energy: torch.nn.Linear = weights_energy
        self.alpha = alpha
        self.device = device
        self.reduction = reduction
        self.nll = cross_entropy

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param logits: logits of shape :math:`B \\times C` or :math:`B \\times C \\times H \\times W`
        :param y: labels of shape :math:`B` or :math:`B \\times H \\times W`; labels :math:`< 0` are OOD
        :return: loss; if the reduction is ``none``, the cross-entropy is per sample (or per pixel) and the
            regularizer is added as a scalar
        """

        regularization = self._regularization(logits, y)
        loss = self.nll(logits, y, reduction=self.reduction)
        return apply_reduction(loss, self.reduction) + apply_reduction(
            self.alpha * regularization, self.reduction
        )

    def _regularization(self, logits, y):
        """

        :param logits: logits
        :param y: labels
        """
        # Permutation depends on shape of logits

        if len(logits.shape) == 4:
            logits_form = logits.permute(0, 2, 3, 1)
        else:
            logits_form = logits

        energy_x_in = self._energy(logits_form[is_known(y)])
        energy_v_out = self._energy(logits_form[is_unknown(y)])

        return self._calculate_reg_loss(energy_x_in, energy_v_out)

    def _calculate_reg_loss(self, energy_score_for_fg, energy_score_for_bg):
        """
        :param energy_score_for_fg: energy score for in-of-distribution samples
        :param energy_score_for_bg: energy score for out-of-distribution samples
        :param features: features of in-of-distribution samples
        :param ood_samples: out-of-distribution samples
        """
        input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
        labels_for_lr = torch.cat(
            (
                torch.ones(len(energy_score_for_fg)).to(self.device),
                torch.zeros(len(energy_score_for_bg)).to(self.device),
            ),
            -1,
        )

        output1 = self.logistic_regression(input_for_lr.view(-1, 1))
        lr_reg_loss = self.nll(output1, labels_for_lr.long())
        return lr_reg_loss

    def _energy(self, logits, dim=1, keepdim=False):
        """
        Numerically stable implementation of the energy calculation
        :param logits: logits
        :param dim: dimension to reduce
        :param keepdim: keep dimension

        """
        m, _ = torch.max(logits, dim=dim, keepdim=True)
        value0 = logits - m
        if keepdim is False:
            m = m.squeeze(dim)
        return -(
            m
            + torch.log(
                torch.sum(
                    F.relu(self.weights_energy.weight) * torch.exp(value0),
                    dim=dim,
                    keepdim=keepdim,
                )
            )
        )


class VirtualOutlierSynthesizingRegLoss(VOSRegLoss):
    """
    Implements the loss function of *VOS: Learning what you don’t know by virtual outlier synthesis* with additional
    sampling of virtual outliers. These outliers are synthesized by fitting a gaussian to the latent features and
    sampling from low-likelihood regions. This alleviates the need for real outliers during training; samples
    with targets :math:`< 0` are ignored.

    For more information see :class:`VOS Energy-Based Loss<pytorch_ood.loss.vos.VOSRegLoss>`.
    """

    info = LossInfo(
        paper=Paper(
            title="VOS: Learning What You Don't Know by Virtual Outlier Synthesis",
            venue="ICLR",
            year=2022,
            url="https://arxiv.org/pdf/2202.01197.pdf",
            code="https://github.com/deeplearning-wisc/vos/",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.LOGITS, Representation.FEATURES},
        supervised=False,
    )

    def __init__(
        self,
        logistic_regression: torch.nn.Linear,
        weights_energy: torch.nn.Linear,
        device: str,
        num_classes: int,
        num_input_last_layer: int,
        fc: torch.nn.Linear,
        alpha: float = 0.1,
        reduction: str = "mean",
        sample_number: int = 1000,
        select: int = 1,
        sample_from: int = 10000,
    ) -> None:
        """
        :param logistic_regression: :math:`\\phi` function. Can be for example a linear layer.
        :param weights_energy: neural network layer, with weights for the energy
        :param device: For example ``cpu`` or ``cuda:0``
        :param num_classes: number of classes
        :param num_input_last_layer: number of inputs in the last layer of the network
        :param fc: fully connected last layer of the network
        :param alpha: weight :math:`\\alpha` of the regularization
        :param reduction: reduction method to apply, one of ``mean``, ``sum`` or ``none``
        :param sample_number: number of ID feature vectors stored per class (queue length); virtual outliers are
            only generated once all queues are full
        :param select: number of lowest-likelihood samples per class that are kept as virtual outliers
        :param sample_from: number of samples drawn from each class-conditional Gaussian, from which the
            ``select`` lowest-likelihood ones are chosen
        """
        super(VirtualOutlierSynthesizingRegLoss, self).__init__(
            logistic_regression,
            weights_energy,
            device=device,
            alpha=alpha,
            reduction=reduction,
        )
        self.num_classes = num_classes
        self.num_input_last_layer = num_input_last_layer
        self.fc = fc
        self.sample_number = sample_number
        self.select = select
        self.sample_from = sample_from

        self.number_dict = {}
        for i in range(num_classes):
            self.number_dict[i] = 0
        self.data_dict = torch.zeros(
            num_classes, self.sample_number, self.num_input_last_layer
        ).to(self.device)
        self.eye_matrix = torch.eye(self.num_input_last_layer, device=self.device)

    def forward(self, logits: torch.Tensor, features: torch.Tensor, y: torch.Tensor):
        """
        Until ``sample_number`` features have been stored for every class, no virtual outliers are
        synthesized and the regularization is zero.

        :param logits: logits of shape :math:`B \\times C`
        :param features: penultimate features of shape :math:`B \\times D`
        :param y: labels of shape :math:`B`; samples with labels :math:`< 0` are discarded
        :return: loss
        :raises NotImplementedError: for segmentation inputs
        """
        if y.dim() != 1:
            raise NotImplementedError("Segmentation not implemented yet")
        y, logits, features = drop_unknown(y, logits, features)
        if len(y) == 0:
            zero = (logits.sum(dim=1) + features.sum(dim=1)) * 0.0
            return zero if self.reduction in (None, "none") else zero.sum()

        regularization = self._regularization(logits, features, y)
        loss = self.nll(logits, y, reduction=self.reduction)
        return apply_reduction(loss, self.reduction) + apply_reduction(
            self.alpha * regularization, self.reduction
        )

    def _regularization(self, prediction, features, target):
        """
        :param prediction: logits
        :param features: features
        :param target: labels
        """
        if len(target.shape) == 3:
            return self._regularization_segmentation(prediction, features, target)
        else:
            return self._regularization_classification(prediction, features, target)

    def _regularization_classification(self, prediction, features, target):
        """
        :param prediction: logits
        :param features: features
        :param target: labels; samples with targets :math:`< 0` are ignored
        """
        known = is_known(target)
        queues_were_full = self._queues_full()
        self._update_queues(features[known], target[known])
        # virtual outliers are only synthesized once the queues were full before this step
        virtual = self._sample_virtual_outliers() if queues_were_full else None
        return self._energy_regularization(prediction, known, virtual)

    def _queues_full(self) -> bool:
        return sum(self.number_dict.values()) == self.num_classes * self.sample_number

    def _update_queues(self, features, target):
        """
        Store the features of ID samples in the per-class queues: fill them up first,
        then replace the oldest entries.
        """
        if not self._queues_full():
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]  # get class id
                if self.number_dict[dict_key] < self.sample_number:
                    self.data_dict[dict_key][self.number_dict[dict_key]] = features[index].detach()
                    self.number_dict[dict_key] += 1
        else:
            for index in range(len(target)):
                dict_key = target[index]
                self.data_dict[dict_key] = torch.cat(
                    (
                        self.data_dict[dict_key][1:],
                        features[index].detach().view(1, -1),
                    ),
                    0,
                )

    def _sample_virtual_outliers(self):
        """
        Fit class-conditional Gaussians with a shared covariance to the queues and keep the
        ``select`` lowest-likelihood of ``sample_from`` samples per class.
        """
        # the covariance finder needs the data to be centered.
        for index in range(self.num_classes):
            if index == 0:
                X = self.data_dict[index] - self.data_dict[index].mean(0)
                mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
            else:
                X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                mean_embed_id = torch.cat(
                    (mean_embed_id, self.data_dict[index].mean(0).view(1, -1)), 0
                )

        # add the variance.
        temp_precision = torch.mm(X.t(), X) / len(X)
        temp_precision += 0.0001 * self.eye_matrix

        # create distributions for each class
        for index in range(self.num_classes):
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embed_id[index], covariance_matrix=temp_precision
            )
            negative_samples = new_dis.rsample((self.sample_from,))
            prob_density = new_dis.log_prob(negative_samples)
            # keep the data in the low density area.
            cur_samples, index_prob = torch.topk(-prob_density, self.select)
            if index == 0:
                ood_samples = negative_samples[index_prob]
            else:
                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
        return ood_samples

    def _energy_regularization(self, prediction, known, virtual):
        """
        Energy regularization between the ID samples of the batch and the virtual outliers;
        samples with targets :math:`< 0` are ignored.
        """
        if virtual is None or len(virtual) == 0 or not known.any():
            return torch.zeros(1).to(self.device)[0]
        energy_out = self._energy(self.fc(virtual), 1)
        return self._calculate_reg_loss(self._energy(prediction[known], 1), energy_out)

    def _regularization_segmentation(self, prediction, features, target):
        """
        :param prediction: logits
        :param features: features
        :param target: labels
        """
        raise NotImplementedError("Segmentation not implemented yet")
