"""Single-image inference: quality gate -> model -> calibrated probabilities -> abstain rule."""
import json

import cv2
import numpy as np

from .labels import SUPERCLASSES
from .metrics import sigmoid
from .preprocess import preprocess_bgr
from .quality import QualityGate

DISCLAIMER = ("Research prototype for academic demonstration. Not a medical device; "
              "not for diagnosis or treatment decisions.")


class PhotoECGPredictor:
    def __init__(self, ckpt_path, calib_path, device=None):
        import torch

        from .model import build_model

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        self.img_hw = (cfg["img_h"], cfg["img_w"])
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        with open(calib_path) as f:
            self.calib = json.load(f)
        self.gate = QualityGate.from_dict(self.calib["quality_gate"])

    def predict_bytes(self, data):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return dict(status="rejected", reasons=["file is not a readable image"], disclaimer=DISCLAIMER)
        return self.predict_image(img)

    def predict_image(self, img_bgr):
        rep = self.gate.check(img_bgr)
        if not rep.ok:
            return dict(status="rejected", reasons=rep.reasons, quality=rep.metrics, disclaimer=DISCLAIMER)
        x = self.torch.from_numpy(preprocess_bgr(img_bgr, self.img_hw))[None].to(self.device)
        with self.torch.no_grad():
            logits = self.model(x).float().cpu().numpy()[0]
        p = sigmoid(logits / self.calib["temperature"])
        p_abn = float(1.0 - p[0])
        conf = max(p_abn, 1.0 - p_abn)
        thr = self.calib["threshold_binary"]
        status = "ok" if conf >= self.calib["tau"] else "abstain"
        return dict(
            status=status,
            probs={c: float(v) for c, v in zip(SUPERCLASSES, p)},
            p_abnormal=p_abn, confidence=float(conf),
            decision=("abnormal" if p_abn >= thr else "normal") if status == "ok" else None,
            threshold=thr, tau=self.calib["tau"], quality=rep.metrics, disclaimer=DISCLAIMER,
        )
