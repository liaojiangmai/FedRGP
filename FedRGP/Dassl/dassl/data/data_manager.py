import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from Dassl.dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
):
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0
    return data_loader


class DataManager:
    """Builds global and federated data loaders."""

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        dataset = build_dataset(cfg)

        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Optional global loaders (some datasets are federated-only).
        self.train_loader_x = None
        if getattr(dataset, "train_x", None):
            self.train_loader_x = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        self.train_loader_u = None
        if getattr(dataset, "train_u", None):
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            self.train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        self.val_loader = None
        if getattr(dataset, "val", None):
            self.val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        self.test_loader = None
        if getattr(dataset, "test", None):
            self.test_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.test,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        # Federated loaders: client_id -> DataLoader
        self.fed_train_loader_x_dict = {}
        self.fed_test_loader_x_dict = {}
        # Per-client base(other) test sets for class-disjoint base-to-novel experiments
        self.fed_test_loader_base_x_dict = {}

        fed_train = getattr(dataset, "federated_train_x", None)
        fed_test = getattr(dataset, "federated_test_x", None)
        fed_test_base = getattr(dataset, "federated_test_base_x", None)
        num_users = getattr(cfg.DATASET, "USERS", 0)
        num_classes = getattr(dataset, "num_classes", 0) or len(getattr(dataset, "classnames", []))
        label_noise_ratio = getattr(cfg.DATASET, "LABEL_NOISE_RATIO", 0.0)
        label_noise_type = getattr(cfg.DATASET, "LABEL_NOISE_TYPE", "symmetric")
        label_noise_seed = getattr(cfg.DATASET, "LABEL_NOISE_SEED", 0)
        label_noise_per_client = getattr(cfg.DATASET, "LABEL_NOISE_PER_CLIENT", True)
        # If the dataset does not explicitly provide a base(other) split, infer it for
        # class-disjoint base-to-novel experiments from the per-client local test splits.
        if (
            fed_test_base is None
            and fed_test is not None
            and getattr(cfg.DATASET, 'PARTITION', '') in ('class-disjoint', 'noniid-class', 'pathological-noniid')
            and getattr(dataset, 'test', None) is not None
            and num_users
        ):
            local_label_sets = {}
            if isinstance(fed_test, dict):
                for cid in range(num_users):
                    items = fed_test.get(cid) or []
                    local_label_sets[cid] = {it.label for it in items}
            elif isinstance(fed_test, (list, tuple)):
                for cid in range(num_users):
                    items = fed_test[cid] if cid < len(fed_test) and fed_test[cid] is not None else []
                    local_label_sets[cid] = {it.label for it in items}

            inferred = {}
            for cid in range(num_users):
                local_labels = local_label_sets.get(cid, set())
                inferred[cid] = [it for it in dataset.test if it.label not in local_labels]
            fed_test_base = inferred


        def _get_client_split(container, client_id):
            if container is None:
                return None
            if isinstance(container, dict):
                return container.get(client_id)
            if isinstance(container, (list, tuple)):
                if 0 <= client_id < len(container):
                    return container[client_id]
                return None
            return None

        def _build_noisy_labels(data_source, rng):
            if not data_source or label_noise_ratio <= 0:
                return None
            labels = [int(getattr(item, "label", 0)) for item in data_source]
            noisy_labels = labels[:]
            for i, y in enumerate(labels):
                if rng.rand() < label_noise_ratio:
                    if label_noise_type == "pairflip":
                        new_y = (y + 1) % num_classes
                    else:
                        new_y = int(rng.randint(0, max(num_classes, 1)))
                        if num_classes > 1 and new_y == y:
                            new_y = (y + 1) % num_classes
                    noisy_labels[i] = new_y
            return noisy_labels

        def _make_noisy_wrapper(noisy_labels):
            class _NoisyWrapper(DatasetWrapper):
                def __init__(self, cfg, data_source, transform=None, is_train=False):
                    super().__init__(cfg, data_source, transform=transform, is_train=is_train)
                    self._noisy_labels = noisy_labels

                def __getitem__(self, idx):
                    output = super().__getitem__(idx)
                    if self.is_train and self._noisy_labels is not None:
                        output["label"] = self._noisy_labels[idx]
                    return output

            return _NoisyWrapper

        if num_users and (fed_train is not None or fed_test is not None):
            for client_id in range(num_users):
                train_split = _get_client_split(fed_train, client_id)
                if train_split:
                    noisy_labels = None
                    wrapper = dataset_wrapper
                    if label_noise_ratio > 0 and num_classes > 0:
                        seed = label_noise_seed + client_id if label_noise_per_client else label_noise_seed
                        rng = np.random.RandomState(seed)
                        noisy_labels = _build_noisy_labels(train_split, rng)
                        if noisy_labels is not None:
                            wrapper = _make_noisy_wrapper(noisy_labels)
                            noise_count = sum(int(a != b) for a, b in zip(noisy_labels, [it.label for it in train_split]))
                            print(f"Client {client_id}: label noise applied ({noise_count}/{len(noisy_labels)} = {noise_count/len(noisy_labels):.2%})")
                    self.fed_train_loader_x_dict[client_id] = build_data_loader(
                        cfg,
                        sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                        data_source=train_split,
                        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                        n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                        n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                        tfm=tfm_train,
                        is_train=True,
                        dataset_wrapper=wrapper,
                    )
                else:
                    self.fed_train_loader_x_dict[client_id] = None

                test_split = _get_client_split(fed_test, client_id)
                if test_split:
                    self.fed_test_loader_x_dict[client_id] = build_data_loader(
                        cfg,
                        sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                        data_source=test_split,
                        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                        tfm=tfm_test,
                        is_train=False,
                        dataset_wrapper=dataset_wrapper,
                    )
                else:
                    self.fed_test_loader_x_dict[client_id] = None

                test_base_split = _get_client_split(fed_test_base, client_id)
                if test_base_split:
                    self.fed_test_loader_base_x_dict[client_id] = build_data_loader(
                        cfg,
                        sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                        data_source=test_base_split,
                        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                        tfm=tfm_test,
                        is_train=False,
                        dataset_wrapper=dataset_wrapper,
                    )
                else:
                    self.fed_test_loader_base_x_dict[client_id] = None

        # Expose dataset metadata
        self.dataset = dataset
        self._num_classes = getattr(dataset, "num_classes", 0)
        self._num_source_domains = len(getattr(cfg.DATASET, "SOURCE_DOMAINS", ()))
        self._lab2cname = getattr(dataset, "lab2cname", {})
        self._classnames = getattr(dataset, "classnames", [])

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    def show_dataset_summary(self, cfg):
        try:
            from tabulate import tabulate
        except Exception:
            tabulate = None

        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}" if getattr(self.dataset, "train_x", None) else "0"])
        if getattr(self.dataset, "train_u", None):
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if getattr(self.dataset, "val", None):
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}" if getattr(self.dataset, "test", None) else "0"])

        if tabulate is None:
            for k, v in table:
                print(f"{k}: {v}")
        else:
            print(tabulate(table))


class DatasetWrapper(TorchDataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train

        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times because transform is None".format(self.k_tfm)
            )

        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode), T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            to_tensor.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        label = getattr(item, "label", 0)
        domain = getattr(item, "domain", 0)
        impath = getattr(item, "impath", "")

        output = {"label": label, "domain": domain, "impath": impath, "index": idx}

        # Support both path-based Datum and array-based Datum (e.g. CIFAR).
        if impath:
            img0 = read_image(impath)
        elif hasattr(item, "data"):
            img0 = Image.fromarray(item.data).convert("RGB")
        else:
            raise ValueError("Dataset item must provide either `impath` or `data`")

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                output["img"] = self._transform_image(self.transform, img0)
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)

        return output

    def _transform_image(self, tfm, img0):
        img_list = []
        for _ in range(self.k_tfm):
            img_list.append(tfm(img0))
        if len(img_list) == 1:
            return img_list[0]
        return img_list
