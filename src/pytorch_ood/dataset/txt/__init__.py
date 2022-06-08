"""
Newsgroups
----------------------------------------------
.. autoclass:: pytorch_ood.dataset.txt.NewsGroup20
   :members:

Reuters 52
----------------------------------------------
.. autoclass:: pytorch_ood.dataset.txt.Reuters52
   :members:

Multi30k
----------------------------------------------
.. autoclass:: pytorch_ood.dataset.txt.Multi30k
   :members:

WMT16 Sentences
----------------------------------------------
.. autoclass:: pytorch_ood.dataset.txt.WMT16Sentences
   :members:

WikiText 2
----------------------------------------------
.. autoclass:: pytorch_ood.dataset.txt.WikiText2
   :members:

WikiText 103
----------------------------------------------
.. autoclass:: pytorch_ood.dataset.txt.WikiText103
   :members:

"""
from .multi30k import Multi30k
from .newsgroups import NewsGroup20
from .reuters import Reuters52, Reuters8
from .wiki import WikiText2, WikiText103
from .wmt16 import WMT16Sentences
