#!/usr/bin/env python3
"""Evaluate a trained early-exit damage classifier under the corruption
pipeline on one or more real held-out crop sets, caching per-exit softmax
outputs and labels to .npz for the downstream risk-coverage / conformal
analysis in analyze_results.py. Every number here comes from an actual
forward pass over actual xBD crops; no result is interpolated.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import EarlyExitResNet18
from corruptions import corrupt, CORRUPTION_TYPES
from train import CLASSES, CLASS_TO_IDX, MEAN, STD

TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def list_samples(root_dir):
    samples = []
    for cls in CLASSES:
        d = os.path.join(root_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".jpg"):
                samples.append((os.path.join(d, fn), CLASS_TO_IDX[cls]))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--corruptions", nargs="+", default=CORRUPTION_TYPES)
    args = ap.parse_args()

    model = EarlyExitResNet18(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    torch.set_num_threads(os.cpu_count() or 4)

    samples = list_samples(args.data_dir)
    labels = np.array([lbl for _, lbl in samples])
    print(f"{args.data_dir}: {len(samples)} samples")

    imgs = [Image.open(p).convert("RGB") for p, _ in samples]

    BATCH = 128
    results = {"labels": labels}
    with torch.no_grad():
        for ctype in args.corruptions:
            sev_list = [0] if ctype == "clean" else args.severities
            for sev in sev_list:
                if ctype == "clean" and sev != 0:
                    continue
                probs1, probs2, probs3 = [], [], []
                for bstart in range(0, len(imgs), BATCH):
                    batch_imgs = imgs[bstart:bstart + BATCH]
                    xb = torch.stack([TF(corrupt(im, ctype, sev)) for im in batch_imgs])
                    e1, e2, e3 = model(xb)
                    probs1.append(F.softmax(e1, dim=1).numpy())
                    probs2.append(F.softmax(e2, dim=1).numpy())
                    probs3.append(F.softmax(e3, dim=1).numpy())
                key = f"{ctype}_s{sev}"
                results[f"{key}_exit1"] = np.concatenate(probs1)
                results[f"{key}_exit2"] = np.concatenate(probs2)
                results[f"{key}_exit3"] = np.concatenate(probs3)
                acc3 = (results[f"{key}_exit3"].argmax(1) == labels).mean()
                print(f"  {key}: exit3 acc = {acc3:.3f}", flush=True)

    np.savez_compressed(args.out, **results)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
