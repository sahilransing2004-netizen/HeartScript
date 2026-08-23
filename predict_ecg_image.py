"""
Run the trained ECG image classifier (normal vs abnormal) on new photos.
"""
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


def build_model(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, transform, image_path, device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
    pred_label = "ABNORMAL" if probs[1] > probs[0] else "NORMAL"
    confidence = probs.max().item()
    return pred_label, confidence, probs.tolist()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, device)
    transform = build_eval_transform(args.img_size)

    for img_path in args.images:
        label, conf, probs = predict(model, transform, img_path, device)
        print(f"{img_path}")
        print(f"  Prediction: {label}  (confidence: {conf:.2%})")
        print(f"  P(normal)={probs[0]:.4f}  P(abnormal)={probs[1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="Paths to ECG image files")
    parser.add_argument("--ckpt", default="checkpoints/ecg_image_classifier_best.pt")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
