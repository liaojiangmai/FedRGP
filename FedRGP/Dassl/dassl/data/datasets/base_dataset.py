import os
import os.path as osp
import random
import tarfile
import zipfile
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

try:
    import gdown
except Exception:  # pragma: no cover
    gdown = None

from Dassl.dassl.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class (adapted for this repo's federated training)."""

    dataset_dir = ""
    domains = []

    def __init__(
        self,
        train_x=None,
        train_u=None,
        val=None,
        test=None,
        federated_train_x=None,
        federated_test_x=None,
    ):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test

        # Federated splits: dict[int, list[Datum]] or list[list[Datum]]
        self._federated_train_x = federated_train_x
        self._federated_test_x = federated_test_x

        # Determine a source list to infer class metadata.
        meta_source = train_x or test
        if meta_source is None and isinstance(federated_train_x, dict) and federated_train_x:
            meta_source = next(iter(federated_train_x.values()))
        if meta_source is None and isinstance(federated_train_x, (list, tuple)) and federated_train_x:
            meta_source = federated_train_x[0]

        self._num_classes = self.get_num_classes(meta_source)
        self._lab2cname, self._classnames = self.get_lab2cname(meta_source)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def federated_train_x(self):
        return self._federated_train_x

    @property
    def federated_test_x(self):
        return self._federated_test_x

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        if not data_source:
            return 0
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        if not data_source:
            return {}, []
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            if gdown is None:
                raise ImportError(
                    "gdown is required to download datasets from Google Drive. "
                    "Please install it (e.g. `pip install gdown`)."
                )
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    sampled_items = random.choices(items, k=num_shots) if repeat else items
                dataset.extend(sampled_items)
            output.append(dataset)

        if len(output) == 1:
            return output[0]
        return output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

    # ---------------------------------------------------------------------
    # Federated helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _split_indices_iid(n_items: int, num_users: int) -> Dict[int, List[int]]:
        idxs = np.random.permutation(n_items)
        splits = np.array_split(idxs, num_users)
        return {i: split.tolist() for i, split in enumerate(splits)}

    @staticmethod
    def _split_indices_labeldir(
        labels: Sequence[int],
        num_users: int,
        beta: float,
        min_require_size: int = 2,
    ) -> Dict[int, List[int]]:
        labels = np.asarray(labels)
        n_classes = int(labels.max()) + 1 if labels.size > 0 else 0

        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_users)]
            for k in range(n_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, num_users))
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                split = np.split(idx_k, proportions)
                for j, idx_j in enumerate(split):
                    idx_batch[j].extend(idx_j.tolist())

            min_size = min(len(idx_j) for idx_j in idx_batch) if idx_batch else 0

        out: Dict[int, List[int]] = {}
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            out[j] = idx_batch[j]
        return out

    def generate_federated_dataset(
        self,
        data_source: Sequence[Datum],
        num_shots: int = -1,
        num_users: int = 10,
        is_iid: bool = False,
        repeat_rate: float = 0.0,
        partition: str = "noniid-labeldir",
        beta: float = 0.3,
    ):
        # num_shots is kept for backwards compatibility (the input list is already prepared).
        if data_source is None:
            return None
        if num_users <= 0:
            return None

        n_items = len(data_source)
        if n_items == 0:
            return {i: [] for i in range(num_users)}

        if is_iid or partition in ("homo", "iid"):
            idx_map = self._split_indices_iid(n_items, num_users)
        elif partition in ("class-disjoint", "noniid-class", "pathological-noniid"):
            # Pathological non-IID: assign non-overlapping classes to different clients.
            labels = np.asarray([item.label for item in data_source], dtype=int)
            uniq = np.unique(labels)
            # Split class ids as evenly as possible among clients.
            class_groups = np.array_split(uniq, num_users)
            # Store for downstream use (e.g., building local/base-other test sets).
            self._client_label_groups = {i: [int(x) for x in g.tolist()] for i, g in enumerate(class_groups)}

            idx_map = {}
            for i, g in enumerate(class_groups):
                if g.size == 0:
                    idx_map[i] = []
                else:
                    idxs = np.where(np.isin(labels, g))[0]
                    idx_map[i] = idxs.tolist()
        else:
            labels = [item.label for item in data_source]
            idx_map = self._split_indices_labeldir(labels, num_users, beta)

        fed_data = defaultdict(list)
        for user_id, idxs in idx_map.items():
            for idx in idxs:
                fed_data[user_id].append(data_source[idx])

            if repeat_rate and repeat_rate > 0 and len(fed_data[user_id]) > 0:
                n_rep = int(round(len(fed_data[user_id]) * float(repeat_rate)))
                if n_rep > 0:
                    fed_data[user_id].extend(random.choices(fed_data[user_id], k=n_rep))

        return fed_data

    def generate_federated_fewshot_dataset(self, *data_sources, **kwargs):
        if len(data_sources) == 1:
            return self.generate_federated_dataset(data_sources[0], **kwargs)
        return [self.generate_federated_dataset(ds, **kwargs) for ds in data_sources]
