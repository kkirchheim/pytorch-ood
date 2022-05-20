import logging
from typing import Iterable, Set

import numpy as np
from torch.utils.data import Dataset, Subset

log = logging.getLogger(__name__)


class OpenSetSimulation(object):
    """
    Base Class for Open Set Simulations.
    """

    @property
    def unique_targets(self) -> Iterable:
        """
        List of all existing classes
        """
        pass

    @property
    def kkc(self) -> Set:
        """
        Known Known Classes
        """
        pass

    @property
    def kuc(self) -> Set:
        """
        Known Unknown Classes
        """
        pass

    @property
    def uuc(self) -> Set:
        """
        Unknown Unknown Classes
        """
        pass

    def train_dataset(self, in_dist=True, out_dist=False) -> Dataset:
        """

        :param in_dist: include IN data
        :param out_dist: include OOD data
        :return:
        """
        pass

    def val_dataset(self, in_dist=True, out_dist=True) -> Dataset:
        """

        :param in_dist: include IN data
        :param out_dist: include OOD data
        :return:
        """
        pass

    def test_dataset(self, in_dist=True, out_dist=True) -> Dataset:
        """

        :param in_dist: include IN data
        :param out_dist: include OOD data
        :return:
        """
        pass


class DynamicOSS(OpenSetSimulation):
    """
    Dynamically samples an Open Set Simulation from a dataset.

    :param train_size: ratio of test samples
    :param val_size: ratio of validation samples
    :param test_size: ratio of test samples
    :param kuc: number of out-of-distribution classes in training set (known unknowns)
    :param uuc_val: number of out-of-distribution classes in validation set (unknown unknowns)
    :param uuc_test: number of out-of-distribution classes in test set (unknown unknowns + test)
    :param seed: seed to use for splits
    """

    def __init__(
        self,
        dataset,
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
        kuc: int = 0,
        uuc_val: int = 2,
        uuc_test: int = 2,
        seed=None,
    ):
        self._dataset = dataset
        self._targets = self.get_targets(self._dataset)
        self._unique_targets = np.unique(self._targets)
        assert type(train_size) is float
        assert type(val_size) is float
        assert type(test_size) is float
        if seed is None:
            self.seed = np.random.randint(0, 1e10)
        else:
            self.seed = seed
        self.r_train = train_size
        self.r_val = val_size
        self.r_test = test_size
        self.out_train = kuc  # known unknowns
        self.out_val = uuc_val
        self.out_test = uuc_test
        self.indices = {
            "train": {"kkc": [], "kuc": [], "uuc": []},
            "val": {"kkc": [], "kuc": [], "uuc": []},
            "test": {"kkc": [], "kuc": [], "uuc": []},
        }
        self.classes = {
            "train": {"kkc": [], "kuc": [], "uuc": []},
            "val": {"kkc": [], "kuc": [], "uuc": []},
            "test": {"kkc": [], "kuc": [], "uuc": []},
        }
        self.class2idx = {}
        # sanity checks
        # assert sum([self.n_train, self.n_val, self.n_test]) == len(self._dataset)
        self._split(self.seed)

    @property
    def kkc(self) -> Set:
        c = set()
        for subset in ["train", "val", "test"]:
            c = c.union(self.classes[subset]["kkc"])
        return c

    @property
    def kuc(self) -> Set:
        c = set()
        for subset in ["train", "val", "test"]:
            c = c.union(self.classes[subset]["kuc"])
        return c

    @property
    def uuc(self) -> Set:
        c = set()
        for subset in ["train", "val", "test"]:
            c = c.union(self.classes[subset]["uuc"])
        return c

    def __repr__(self):
        return f"DynamicOSSim({self._dataset}, seed={self.seed})"

    def _get_subset(self, stage, in_dist, out_dist):
        indices = []
        if in_dist:
            indices.extend(self.indices[stage]["kkc"])
        if out_dist:
            indices.extend(self.indices[stage]["kuc"])
            indices.extend(self.indices[stage]["uuc"])
        return Subset(self._dataset, indices)

    def train_dataset(self, in_dist: bool = True, out_dist: bool = True) -> Subset:
        return self._get_subset("train", in_dist, out_dist)

    def val_dataset(self, in_dist: bool = True, out_dist: bool = True) -> Subset:
        return self._get_subset("val", in_dist, out_dist)

    def test_dataset(self, in_dist: bool = True, out_dist: bool = True) -> Subset:
        return self._get_subset("test", in_dist, out_dist)

    @property
    def unique_targets(self):
        return self._unique_targets

    def _split(self, seed):
        # make sure results are stable
        rng: np.random.Generator = np.random.default_rng(seed=seed)
        log.debug(
            f"Creating ossim for dataset with {len(self._dataset)} samples and {len(self.unique_targets)} classes"
        )
        # split classes
        perm_class = rng.permutation(self.unique_targets)
        train_out_c = perm_class[: self.out_train]
        val_out_c = perm_class[self.out_train : self.out_train + self.out_val]
        test_out_c = perm_class[
            self.out_train + self.out_val : self.out_train + self.out_val + self.out_test
        ]
        train_in_c = perm_class[self.out_train + self.out_val + self.out_test :]
        log.debug(f"KKC ({len(train_in_c)}): {train_in_c}")
        log.debug(f"KUC ({len(train_out_c)}): {train_out_c}")
        log.debug(f"UUC [test] ({len(test_out_c)}): {test_out_c}")
        log.debug(f"UUC [val] ({len(val_out_c)}): {val_out_c}")
        # get indexes of samples for each class
        for target in self.unique_targets:
            idx = np.arange(len(self._dataset))[self._targets == target]
            self.class2idx[target] = idx
            log.debug(f"{target} -> {len(idx)}")
        # split samples into sets
        # known known classes -> train, val and test
        for clazz in train_in_c:
            perm_idx = rng.permutation(self.class2idx[clazz])
            # log.debug(f"kkc {clazz} -> {perm_idx}")
            n_train = int(len(perm_idx) * self.r_train)
            n_val = int(len(perm_idx) * self.r_val)
            self.indices["train"]["kkc"].extend(perm_idx[:n_train])
            self.indices["val"]["kkc"].extend(perm_idx[n_train : n_train + n_val])
            self.indices["test"]["kkc"].extend(perm_idx[n_train + n_val :])
            for stage in ["train", "val", "test"]:
                self.classes[stage]["kkc"].append(clazz)
                self.indices[stage]["kkc"].sort()
        # known unknown classes  -> train, val and test
        for clazz in train_out_c:
            perm_idx = rng.permutation(self.class2idx[clazz])
            log.debug(f"kuc {clazz} -> {perm_idx}")
            n_train = int(len(perm_idx) * self.r_train)
            n_val = int(len(perm_idx) * self.r_val)
            self.indices["train"]["kuc"].extend(perm_idx[:n_train])
            self.indices["val"]["kuc"].extend(perm_idx[n_train : n_train + n_val])
            self.indices["test"]["kuc"].extend(perm_idx[n_train + n_val :])
            for stage in ["train", "val", "test"]:
                self.classes[stage]["kuc"].append(clazz)
                self.indices[stage]["kuc"].sort()
        # unknown unknown classes -> val and test respectively
        for clazz in val_out_c:
            idx = self.class2idx[clazz]
            self.indices["val"]["uuc"].extend(idx)
            self.indices["val"]["kuc"].sort()
            self.classes["val"]["uuc"].append(clazz)
        for clazz in test_out_c:
            idx = self.class2idx[clazz]
            self.indices["test"]["uuc"].extend(idx)
            self.indices["test"]["kuc"].sort()
            self.classes["test"]["uuc"].append(clazz)

    @staticmethod
    def get_targets(dataset):
        """
        For some datasets, this might be an expensive operation
        """
        if hasattr(dataset, "targets"):
            return dataset.targets

        # this might be slow
        return [target for _, target in dataset]
