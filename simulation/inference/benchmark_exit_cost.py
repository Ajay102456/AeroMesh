#!/usr/bin/env python3
"""Measure real CPU wall-clock inference cost per early-exit depth, so the
manuscript's early-exit compute-savings claim is backed by an actual
timed measurement on this hardware rather than only the cited 2x-5x
BranchyNet literature figure.
"""
import time
import torch
from model import EarlyExitResNet18

torch.manual_seed(0)
torch.set_num_threads(4)
model = EarlyExitResNet18(pretrained=False)
model.eval()

BATCH = 1  # a single UAV performing on-tile inference, not a batched server
N_REPEATS = 100
x = torch.randn(BATCH, 3, 112, 112)

with torch.no_grad():
    for _ in range(5):
        model(x)  # warmup

    t_exit1, t_exit2, t_exit3 = [], [], []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        h = model.stem(x)
        h = model.layer1(h)
        h = model.layer2(h)
        _ = model.exit1_head(model.pool(h))
        t_exit1.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        h2 = model.stem(x)
        h2 = model.layer1(h2)
        h2 = model.layer2(h2)
        h2 = model.layer3(h2)
        _ = model.exit2_head(model.pool(h2))
        t_exit2.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = model(x)
        t_exit3.append(time.perf_counter() - t0)

import statistics as st
print(f"exit1 mean latency: {st.mean(t_exit1)*1000:.3f} ms (std {st.stdev(t_exit1)*1000:.3f})")
print(f"exit2 mean latency: {st.mean(t_exit2)*1000:.3f} ms (std {st.stdev(t_exit2)*1000:.3f})")
print(f"exit3 mean latency: {st.mean(t_exit3)*1000:.3f} ms (std {st.stdev(t_exit3)*1000:.3f})")
print(f"speedup exit1 vs exit3: {st.mean(t_exit3)/st.mean(t_exit1):.2f}x")
print(f"speedup exit2 vs exit3: {st.mean(t_exit3)/st.mean(t_exit2):.2f}x")
