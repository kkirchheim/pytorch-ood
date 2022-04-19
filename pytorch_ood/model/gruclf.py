"""
Text classifier used by Hendrycks et al.
"""
from torch import nn
from torch.hub import load_state_dict_from_url


class GRUClassifier(nn.Module):
    """
    Classifier with token embedding and multi layer gated recurrent unit (GRU) for text classification.

    :see Implementation: https://github.com/hendrycks/outlier-exposure/blob/master/NLP_classification/train.py
    """

    def __init__(self, num_classes, n_vocab, embedding_dim=50):
        """

        :param num_classes:
        :param n_vocab:
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

    def forward(self, x):
        embeds = self.embedding(x)
        z = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.linear(z)
        return logits

    @staticmethod
    def from_pretrained(name, **kwargs):
        """

        :param name:
        :param kwargs:
        :return:
        """
        urls = {
            "20ng-base": "https://github.com/hendrycks/outlier-exposure/raw/master/NLP_classification/snapshots/20ng/baseline/model.dict",
            "20ng-oe-wiki2": "https://github.com/hendrycks/outlier-exposure/raw/master/NLP_classification/snapshots/20ng/OE/wikitext2/model_finetune.dict",
        }

        url = urls[name]
        model = GRUClassifier(**kwargs)
        state_dict = load_state_dict_from_url(url, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)
        return model
