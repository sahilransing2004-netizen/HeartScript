# HeartScript

ECG anomaly detection on PTB-XL using a 3-stage pipeline:
**MAE pretraining → SimCLR contrastive tuning → Prototypical Networks few-shot classification.**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Get the data

```bash
python data/download_ptbxl.py --root ./ptbxl_data
```

This pulls the **100Hz** version of PTB-XL (~1.7GB) via wget-mirroring the PhysioNet
directory. If your network blocks `physionet.org`, download manually from
https://physionet.org/content/ptb-xl/1.0.3/ and drop it into `./ptbxl_data`.

Verify you have `ptbxl_database.csv` and `scp_statements.csv` directly under `ptbxl_data/`.

## 2. Sanity check (no data needed)

```bash
python3 smoke_test.py
```

Runs all three models on random tensors — confirms shapes and losses compute
correctly before you burn time on real training.

## 3. Run the pipeline, in order

```bash
python train_mae.py --config configs/config.yaml
python train_simclr.py --config configs/config.yaml --init_from checkpoints/mae_encoder.pt
python train_protonet.py --config configs/config.yaml --init_from checkpoints/simclr_encoder.pt
```

Each stage saves its encoder to `checkpoints/` and the next stage loads it automatically.

## Config knobs that matter on your machine (TITAN, no/weak GPU)

All in `configs/config.yaml`:
- `data.subset_fraction: 0.15` — use 15% of PTB-XL for fast iteration. Bump to `1.0`
  for the real run once the pipeline is validated.
- `*.device: "cpu"` — flip to `"cuda"` per-stage once you're on a GPU box (Colab/AWS).
- `*.batch_size` — MAE/SimCLR are set conservatively (32/64) for CPU. Lower further
  if you hit memory issues, raise a lot once on GPU.
- `encoder.type: "cnn1d"` — don't switch to `"transformer"` until you have a real GPU;
  the current code only implements the CNN path.

## Current label setup

Labels are collapsed to **binary**: `NORM` (normal) vs everything else (abnormal),
using PTB-XL's `diagnostic_superclass` aggregation. This is intentional for stage 3 —
2-way, 5-shot classification is a sane starting point. Once that works, extend
`protonet.n_way` and the label logic in `data/dataset.py` to the 5 PTB-XL superclasses
(NORM, MI, STTC, CD, HYP) for a harder, more useful multi-class version.


## Heart risk demo: folder layout
- `backend/` is what the server runs: `app.py`, the model code in `backend/risk/`, and the web page in `backend/static/index.html`.
- `risk/` is the working folder for the heart risk model: training and evaluation scripts, `heart.csv` and `results.csv`.
- After changing `risk/predict_risk.py`, copy it to `backend/risk/predict_risk.py`. The server only reads the `backend/` copy.
- Start the server from `backend/`: `python -m uvicorn app:app --port 8000`, then open http://127.0.0.1:8000.
## What's NOT done yet

- No hyperparameter tuning — these are reasonable starting defaults, not tuned.
- No test-set evaluation script yet (train_protonet.py only does train/val).
- No experiment tracking (add W&B or plain CSV logging once you're iterating for real).
- Transformer encoder variant not implemented (CNN only).
- Multi-class (5-way) PTB-XL classification not wired up — binary only right now.
