"""
Text classifier used in the Outlier Exposure experiments of Hendrycks et al.
"""

import torch
from torch import nn


class GRUClassifier(nn.Module):
    """
    Classifier with token embedding and multi layer gated recurrent unit (GRU) for text classification,
    as used in the NLP experiments of
    `Deep Anomaly Detection with Outlier Exposure <https://arxiv.org/abs/1812.04606>`__
    (Hendrycks et al.).

    The GRU has two layers and a hidden size of 128, and the embedding uses index 1 as
    padding token. Features are the final hidden state of the last layer, of dimension 128.

    :see Implementation:
        `GitHub <https://github.com/hendrycks/outlier-exposure/blob/master/NLP_classification/train.py>`__
    """

    def __init__(self, num_classes: int, n_vocab: int, embedding_dim: int = 50):
        """
        :param num_classes: number of classes in the dataset
        :param n_vocab: size of the vocabulary, i.e. number of distinct tokens
        :param embedding_dim: embedding size
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=1)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(128, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: token indices of shape :math:`B \\times L`, dtype ``long``
        :return: features of shape :math:`B \\times 128`
        """
        embeds = self.embedding(x)
        return self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: token indices of shape :math:`B \\times L`, dtype ``long``
        :return: logits of shape :math:`B \\times K`
        """
        embeds = self.embedding(x)
        z = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.fc(z)
        return logits
