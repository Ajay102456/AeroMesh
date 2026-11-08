#!/usr/bin/env python3
"""Real joint ns-3 validation figure for the Philippines scenario (see
joint_analyze.py). Two panels: (a) real time-resolved Phi(t), full vs
no_self_healing, across the three failure severities; (b) real per-replicate
composite score, full vs fixed_roles, the same comparison Table B of the
Python co-simulator reports at reduced fidelity for all three scenarios.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/home/s/Documents/AeroMesh/manuscript/figures"
EPOCH_S = 30.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

fa = json.load(open("joint_fig_a_ph.json"))
t = [(_ + 1) * EPOCH_S for _ in range(10)]
colors = {0: "#4C72B0", 1: "#DD8452", 2: "#C44E52"}
for fl in (0, 1, 2):
    full = next(d for d in fa if d["failLevel"] == fl and d["ablation"] == "full")
    nosh = next(d for d in fa if d["failLevel"] == fl and d["ablation"] == "no_self_healing")
    ax1.plot(t, full["phi_t"], "-", color=colors[fl], label=f"AeroMesh, failLevel {fl}")
    ax1.plot(t, nosh["phi_t"], "--", color=colors[fl], alpha=0.6, label=f"No self-healing, failLevel {fl}")
ax1.set_xlabel("Mission time (s)")
ax1.set_ylabel(r"Priority-weighted connectivity $\Phi(t)$")
ax1.set_ylim(0.85, 1.02)
ax1.set_title("(a) Real joint ns-3, Philippines", fontsize=9)
ax1.legend(fontsize=6, loc="lower right")
ax1.grid(alpha=0.3)

tb = json.load(open("joint_table_b_ph.json"))
rng = np.random.default_rng(1)
full_jit = rng.normal(0, 0.03, tb["n_pairs"])
fr_jit = rng.normal(0, 0.03, tb["n_pairs"])
ax2.scatter(0 + full_jit, [tb["full_mean"]] * tb["n_pairs"], alpha=0)  # keep axis scale sane
ax2.bar(["AeroMesh\n(auction)", "Fixed roles\n(no auction)"],
        [tb["full_mean"], tb["fixed_roles_mean"]],
        color=["#4C72B0", "#937860"], width=0.5)
ax2.set_ylabel("Mean composite score (real ph data)")
ax2.set_title(f"(b) Real ns-3, full vs fixed roles\nn={tb['n_pairs']} pairs, "
              f"Cliff's $\\delta$={tb['cliffs_delta']:.2f}, p={tb['wilcoxon_p']:.3f}", fontsize=9)
ax2.set_ylim(0, max(tb["full_mean"], tb["fixed_roles_mean"]) * 1.2)
ax2.grid(alpha=0.3, axis="y")

fig.suptitle("Figure B-validate. Real joint ns-3 check (Philippines only; see Limitations for scope)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/joint_ph_validation.pdf")
fig.savefig(f"{OUT_DIR}/joint_ph_validation.svg")
print("wrote joint_ph_validation.pdf/svg")
