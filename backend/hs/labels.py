"""PTB-XL metadata, superclass labels and the official fold-based split."""
import ast
import os

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ABNORMAL_CLASSES = ["MI", "STTC", "CD", "HYP"]


def load_ptbxl_metadata(ptbxl_dir, sampling_rate=100):
    """Return a DataFrame (index ecg_id) with 5 superclass columns, 'abnormal',
    'record_path' and 'split' (train: folds 1-8, val: fold 9, test: fold 10).
    Records with no diagnostic superclass are dropped (standard practice)."""
    db_path = os.path.join(ptbxl_dir, "ptbxl_database.csv")
    scp_path = os.path.join(ptbxl_dir, "scp_statements.csv")
    for p in (db_path, scp_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Point --ptbxl_dir at the folder that contains "
                "ptbxl_database.csv, scp_statements.csv and records100/."
            )
    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(scp_path, index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    code_to_class = scp["diagnostic_class"].to_dict()

    def to_super(codes):
        return sorted({code_to_class[c] for c in codes if c in code_to_class})

    df["superclasses"] = df["scp_codes"].apply(to_super)
    df = df[df["superclasses"].map(len) > 0].copy()
    for c in SUPERCLASSES:
        df[c] = df["superclasses"].apply(lambda s, c=c: int(c in s))
    df["abnormal"] = df[ABNORMAL_CLASSES].max(axis=1)
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    df["record_path"] = df[col]
    df["split"] = np.where(
        df["strat_fold"] <= 8, "train", np.where(df["strat_fold"] == 9, "val", "test")
    )
    return df


def load_signal(ptbxl_dir, record_path):
    """Load one record as float32 array (T, 12) in mV."""
    import wfdb

    sig, _ = wfdb.rdsamp(os.path.join(ptbxl_dir, record_path))
    return sig.astype(np.float32)


def sample_label_fraction(df_train, fraction, seed=0, min_pos=5, tries=200):
    """Pick a random subset of training records for the label-efficiency study.
    Retries so that every class has at least `min_pos` positives when possible."""
    ids_all = df_train.index.to_numpy()
    if fraction >= 1.0:
        return ids_all
    n = max(int(round(len(ids_all) * fraction)), 1)
    rng = np.random.default_rng(seed)
    best, best_min = None, -1
    for _ in range(tries):
        ids = rng.choice(ids_all, size=n, replace=False)
        m = int(df_train.loc[ids, SUPERCLASSES].sum().min())
        if m > best_min:
            best, best_min = ids, m
        if m >= min_pos:
            break
    if best_min < min_pos:
        print(f"[warn] label_fraction={fraction}: rarest class has only {best_min} positives")
    return best
