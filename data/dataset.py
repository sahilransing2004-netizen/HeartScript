"""
PyTorch Dataset for PTB-XL.

Reads ptbxl_database.csv for metadata/labels, loads waveform records with wfdb,
crops to fixed-length windows, and z-normalizes per lead.
"""
import ast
import os

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset


# PTB-XL "superclass" diagnostic labels -> collapse to binary normal/abnormal.
# (scp_statements.csv maps individual codes to these superclasses)
NORMAL_SUPERCLASS = "NORM"


class PTBXLDataset(Dataset):
    def __init__(
        self,
        root: str,
        sampling_rate: int = 100,
        window_seconds: int = 5,
        subset_fraction: float = 1.0,
        split: str = "train",  # "train" | "val" | "test"
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
        return_labels: bool = True,
    ):
        self.root = root
        self.sampling_rate = sampling_rate
        self.window_len = window_seconds * sampling_rate
        self.return_labels = return_labels

        db_path = os.path.join(root, "ptbxl_database.csv")
        agg_path = os.path.join(root, "scp_statements.csv")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Couldn't find {db_path}. Run data/download_ptbxl.py first, "
                f"or check --root points at the folder containing ptbxl_database.csv"
            )

        df = pd.read_csv(db_path, index_col="ecg_id")
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)

        agg_df = pd.read_csv(agg_path, index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(scp_codes):
            classes = []
            for code in scp_codes.keys():
                if code in agg_df.index:
                    classes.append(agg_df.loc[code].diagnostic_class)
            return list(set(classes))

        df["diagnostic_superclass"] = df.scp_codes.apply(aggregate_diagnostic)
        df["label"] = df.diagnostic_superclass.apply(
            lambda classes: 0 if classes == [NORMAL_SUPERCLASS] else 1
        )

        # deterministic split
        rng = np.random.RandomState(seed)
        idx = df.index.to_numpy().copy()
        rng.shuffle(idx)

        if subset_fraction < 1.0:
            idx = idx[: int(len(idx) * subset_fraction)]

        n = len(idx)
        n_val = int(n * val_fraction)
        n_test = int(n * test_fraction)
        n_train = n - n_val - n_test

        if split == "train":
            self.ids = idx[:n_train]
        elif split == "val":
            self.ids = idx[n_train : n_train + n_val]
        elif split == "test":
            self.ids = idx[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df = df.loc[self.ids]
        filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.filenames = self.df[filename_col].to_list()
        self.labels = self.df["label"].to_list()

    def __len__(self):
        return len(self.ids)

    def _load_record(self, filename):
        record_path = os.path.join(self.root, filename)
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal.astype(np.float32)  # shape (time, n_leads)
        return sig

    def _crop_and_normalize(self, sig):
        # sig: (time, n_leads) -> center crop/pad to window_len, then transpose to (leads, time)
        t = sig.shape[0]
        if t >= self.window_len:
            start = (t - self.window_len) // 2
            sig = sig[start : start + self.window_len]
        else:
            pad = self.window_len - t
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="constant")

        sig = sig.T  # (leads, time)
        mean = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - mean) / std
        return sig

    def __getitem__(self, i):
        filename = self.filenames[i]
        sig = self._load_record(filename)
        sig = self._crop_and_normalize(sig)

        if self.return_labels:
            return sig, self.labels[i]
        return sig
