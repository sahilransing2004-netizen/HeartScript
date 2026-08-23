"""
Downloads PTB-XL (100Hz version) from PhysioNet.

PTB-XL is ~1.7GB at 100Hz (the 500Hz version is ~30GB — skip that for local work).

Usage:
    python data/download_ptbxl.py --root ./ptbxl_data
"""
import argparse
import os
import subprocess
import sys

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


def download(root: str):
    os.makedirs(root, exist_ok=True)
    # wget mirrors the whole PhysioNet dir tree; -r recursive, -N only-new,
    # -np no-parent, -q quiet-ish. This grabs metadata CSVs + the 100Hz records.
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "-P", root,
        PTBXL_URL,
    ]
    print("Running:", " ".join(cmd))
    print("This can take a while (~1.7GB). If wget isn't allowed on your network,")
    print("download manually from https://physionet.org/content/ptb-xl/1.0.3/")
    print("and place it under:", root)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("wget not found. Install it (`sudo apt install wget`) or download manually.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print("If this is a network/firewall issue, download manually from the URL above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./ptbxl_data")
    args = parser.parse_args()
    download(args.root)
