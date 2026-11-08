#!/usr/bin/env python3
"""Train the early-exit ResNet-18 damage classifier on real xBD building
crops. Supports the forward-transfer direction (train on the non-Palu
United States/Caribbean-centric pool, defined by --train_dir=crops/train)
and the reverse-transfer direction (train on crops/palu_train) named in
the manuscript's cross-dataset transfer protocol.
"""
import argparse
import os
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import EarlyExitResNet18, NUM_CLASSES

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CropDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.samples = []
        for cls in CLASSES:
            d = os.path.join(root_dir, cls)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".jpg"):
                    self.samples.append((os.path.join(d, fn), CLASS_TO_IDX[cls]))
        if train:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


def class_weights(dataset):
    counts = Counter(lbl for _, lbl in dataset.samples)
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        n = counts.get(c, 1)
        weights[c] = total / (NUM_CLASSES * n)
    return weights


def evaluate(model, loader, criterion_weights):
    model.eval()
    correct = [0, 0, 0]
    total = 0
    with torch.no_grad():
        for x, y in loader:
            e1, e2, e3 = model(x)
            for i, e in enumerate([e1, e2, e3]):
                correct[i] += (e.argmax(1) == y).sum().item()
            total += y.size(0)
    return [c / total for c in correct]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(42)

    train_ds = CropDataset(args.train_dir, train=True)
    val_ds = CropDataset(args.val_dir, train=False)
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")
    print("train class counts:", Counter(lbl for _, lbl in train_ds.samples))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = EarlyExitResNet18(pretrained=True)
    weights = class_weights(train_ds)
    print("class weights:", weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    exit_loss_weights = [0.3, 0.4, 1.0]

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            e1, e2, e3 = model(x)
            loss = sum(w * criterion(e, y) for w, e in zip(exit_loss_weights, [e1, e2, e3]))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        accs = evaluate(model, val_loader, weights)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} loss={running_loss/len(train_ds):.4f} "
              f"val_acc(exit1,2,3)={accs} elapsed={elapsed:.0f}s", flush=True)
        if accs[2] > best_acc:
            best_acc = accs[2]
            torch.save(model.state_dict(), args.out)
            print(f"  saved checkpoint to {args.out} (val_acc exit3 = {best_acc:.4f})")

    print(f"Done. Best final-exit val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
