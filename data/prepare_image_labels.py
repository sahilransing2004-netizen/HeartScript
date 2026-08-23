"""
Build a normal/abnormal label CSV for the PTB-XL ECG image dataset.
"""
import argparse
import ast
import os

import pandas as pd


def is_normal(scp_codes_str: str, threshold: float = 80.0) -> bool:
    try:
        codes = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return False
    norm_conf = codes.get("NORM", 0.0)
    return norm_conf >= threshold


def find_image_path(images_root: str, ecg_id: int):
    bucket = f"{(ecg_id // 1000) * 1000:05d}"
    candidate = os.path.join(images_root, bucket, f"{ecg_id:05d}_lr-0.png")
    if os.path.isfile(candidate):
        return candidate
    return None


def main(args):
    df = pd.read_csv(args.ptbxl_csv, index_col="ecg_id")

    rows = []
    missing = 0
    for ecg_id, row in df.iterrows():
        img_path = find_image_path(args.images_root, int(ecg_id))
        if img_path is None:
            missing += 1
            continue
        label = 0 if is_normal(row["scp_codes"]) else 1
        rows.append({"ecg_id": ecg_id, "image_path": img_path, "label": label})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    n_normal = (out_df["label"] == 0).sum()
    n_abnormal = (out_df["label"] == 1).sum()
    print(f"Total matched images: {len(out_df)}")
    print(f"  Normal (0):   {n_normal}")
    print(f"  Abnormal (1): {n_abnormal}")
    print(f"Missing image files (skipped): {missing}")
    print(f"Saved labels -> {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_csv", default="ptbxl_data/ptbxl_database.csv")
    parser.add_argument("--images_root", default="ecg_images_data")
    parser.add_argument("--out_csv", default="ecg_images_data/labels.csv")
    args = parser.parse_args()
    main(args)
