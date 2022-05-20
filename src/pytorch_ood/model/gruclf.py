"""
Text classifier used by Hendrycks et al.
"""
import torch
from torch import nn


class GRUClassifier(nn.Module):
    """
    Classifier with token embedding and multi layer gated recurrent unit (GRU) for text classification.

    :see Implementation: https://github.com/hendrycks/outlier-exposure/blob/master/NLP_classification/train.py
    """

    def __init__(self, num_classes, n_vocab, embedding_dim=50):
        """

        :param num_classes: number of classes in the dataset
        :param n_vocab: size of the vocabulary, i.e. number of distinct tokens
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=1)
        self.gru = nn.GRU(
            input_size=50,
            hidden_size=128,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param: lists of list of tokens

        :returns: logits
        """
        embeds = self.embedding(x)
        z = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.linear(z)
        return logits
