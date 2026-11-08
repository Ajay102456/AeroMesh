#!/usr/bin/env python3
"""Analyze the real joint ns-3 campaign (Philippines scenario only; see
joint_campaign.py for why id/vn are excluded). Produces:

  joint_fig_a_ph.json    - real time-resolved Phi(t), full vs no_self_healing
  joint_table_a_ph.json  - real time-to-restore, full vs no_self_healing
  joint_table_b_ph.json  - real paired comparison, full vs fixed_roles

Phi(t) here is a genuine ns-3 FlowMonitor measurement (not bootstrap-
resampled or approximated): aeromesh_joint_sim.cc actually withdraws UAVs
from relay duty for inference windows and snapshots real packet counts at
every epoch boundary. Psi(t) reuses the same real per-severity inference
accuracy curves used throughout the manuscript (simulation/inference's
analysis_*.json), combined via the same declared role-count formula as
simulation/cosim/cosim.py, so the composite score mixes two independently
real-measured layers through one declared (not measured) combination rule,
consistent with how the rest of the manuscript already reports this split.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cosim"))
from data import load_inference_curves  # noqa: E402

RESULTS_DIR = "results/joint"
CLASSES = ["responder", "medical", "civilian"]
W_PRIORITY = {"responder": 10.0, "medical": 3.0, "civilian": 1.0}
SEVERITY_TO_CORRUPTION = {0: 0, 1: 2, 2: 4}
N_INFERENCE_BASE = 5
MAX_EXTRA_RELAY = 3
EPOCH_S = 30.0


def severity_schedule(start_fail, self_healing, n_epochs=10):
    s = start_fail
    sched = []
    for e in range(n_epochs):
        if self_healing and e > 0 and e % 4 == 0:
            s = max(s - 1, 0)
        sched.append(s)
    return sched


def load_runs():
    runs = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
        base = os.path.basename(path)
        if "campaign_log" in base:
            continue
        parts = base.replace(".csv", "").split("_")
        # tag format: {ablation}_ph_f{failLevel}_s{seed}
        seed = int(parts[-1][1:])
        failLevel = int(parts[-2][1:])
        ablation = "_".join(parts[:-3])
        rows = list(csv.DictReader(open(path)))
        by_epoch = defaultdict(dict)
        for r in rows:
            by_epoch[int(r["epoch"])][r["priorityClass"]] = float(r["pdr"])
        n_epochs = max(by_epoch)
        phi_class_series = {c: [by_epoch[e].get(c, 0.0) for e in range(1, n_epochs + 1)] for c in CLASSES}
        runs.append({"ablation": ablation, "failLevel": failLevel, "seed": seed,
                      "phi_class": phi_class_series})
    return runs


def compute_phi_psi(run, curves):
    self_healing = run["ablation"] != "no_self_healing"
    sched = severity_schedule(run["failLevel"], self_healing)
    phi_series = []
    for e, sev in enumerate(sched):
        num = sum(W_PRIORITY[c] * run["phi_class"][c][e] for c in CLASSES)
        den = sum(W_PRIORITY.values())
        phi_series.append(num / den)

    if run["ablation"] == "fixed_roles":
        acc_curve = curves["risk_coverage_full_conformal"]
        n_inf_series = [N_INFERENCE_BASE] * len(sched)
    else:
        acc_curve = curves["risk_coverage_full_conformal"]
        n_inf_series = [max(N_INFERENCE_BASE - sev, 2) for sev in sched]
    psi_series = []
    for sev, n_inf in zip(sched, n_inf_series):
        corruption = SEVERITY_TO_CORRUPTION[sev]
        acc = acc_curve.get(corruption, acc_curve[max(acc_curve)])
        psi_series.append(n_inf * acc)
    return phi_series, psi_series


def time_to_restore(series, target, tol=0.9):
    for e, v in enumerate(series):
        if v >= tol * target:
            return (e + 1) * EPOCH_S
    return None


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def paired_bootstrap_ci(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diffs.mean(), lo, hi


def main():
    runs = load_runs()
    curves = load_inference_curves(os.path.join(os.path.dirname(__file__), "..", "inference",
                                                  "analysis_nonpalu_indomain.json"))
    for r in runs:
        r["phi_series"], r["psi_series"] = compute_phi_psi(r, curves)
        r["phi_mean"] = float(np.mean(r["phi_series"]))
        r["psi_mean"] = float(np.mean(r["psi_series"]))

    # ---- Figure A: mean Phi(t), full vs no_self_healing, per failLevel ----
    fig_a = []
    for failLevel in (0, 1, 2):
        for ablation in ("full", "no_self_healing"):
            series = [r["phi_series"] for r in runs if r["ablation"] == ablation and r["failLevel"] == failLevel]
            fig_a.append({"scenario": "ph", "failLevel": failLevel, "ablation": ablation,
                           "phi_t": np.mean(series, axis=0).tolist(), "n_seeds": len(series)})
    json.dump(fig_a, open("joint_fig_a_ph.json", "w"), indent=1)

    # ---- Table A: time-to-restore, full vs no_self_healing ----
    target = {}
    for c in CLASSES:
        vals = [r["phi_class"][c][0] for r in runs if r["ablation"] == "full" and r["failLevel"] == 0]
        target[c] = float(np.mean(vals))
    table_a = []
    for failLevel in (0, 1, 2):
        for ablation in ("full", "no_self_healing"):
            for c in CLASSES:
                vals = [time_to_restore(r["phi_class"][c], target[c])
                        for r in runs if r["ablation"] == ablation and r["failLevel"] == failLevel]
                n_restored = sum(1 for v in vals if v is not None)
                restored = [v for v in vals if v is not None]
                table_a.append({"scenario": "ph", "failLevel": failLevel, "ablation": ablation,
                                 "priority_class": c, "n_runs": len(vals), "n_restored": n_restored,
                                 "mean_restore_s": float(np.mean(restored)) if restored else None})
    json.dump(table_a, open("joint_table_a_ph.json", "w"), indent=1)

    # ---- Table B: full vs fixed_roles, paired by (failLevel, seed) ----
    by_key = defaultdict(dict)
    for r in runs:
        by_key[(r["failLevel"], r["seed"])][r["ablation"]] = r

    psi_norm = {}
    for failLevel in (0, 1, 2):
        vals = [r["psi_mean"] for r in runs if r["ablation"] == "full" and r["failLevel"] == failLevel]
        psi_norm[failLevel] = float(np.mean(vals))

    full_vals, fr_vals = [], []
    for (failLevel, seed), d in by_key.items():
        if "full" not in d or "fixed_roles" not in d:
            continue
        full_vals.append(0.5 * d["full"]["phi_mean"] + 0.5 * (d["full"]["psi_mean"] / psi_norm[failLevel]))
        fr_vals.append(0.5 * d["fixed_roles"]["phi_mean"] + 0.5 * (d["fixed_roles"]["psi_mean"] / psi_norm[failLevel]))
    diffs = np.array(full_vals) - np.array(fr_vals)
    mean_diff, lo, hi = paired_bootstrap_ci(diffs)
    try:
        wstat, pval = stats.wilcoxon(full_vals, fr_vals)
    except ValueError:
        pval = 1.0
    delta = cliffs_delta(full_vals, fr_vals)
    table_b = {"scenario": "ph", "ablation": "fixed_roles", "n_pairs": len(diffs),
               "mean_diff": float(mean_diff), "ci_lo": float(lo), "ci_hi": float(hi),
               "wilcoxon_p": float(pval), "cliffs_delta": float(delta),
               "full_mean": float(np.mean(full_vals)), "fixed_roles_mean": float(np.mean(fr_vals))}
    json.dump(table_b, open("joint_table_b_ph.json", "w"), indent=1)

    print("Figure A rows:", len(fig_a))
    print("Table A rows:", len(table_a))
    print("Table B (real ph, full vs fixed_roles):")
    print(f"  n_pairs={table_b['n_pairs']} full_mean={table_b['full_mean']:.4f} "
          f"fixed_roles_mean={table_b['fixed_roles_mean']:.4f}")
    print(f"  diff={table_b['mean_diff']:.4f} CI=[{table_b['ci_lo']:.4f},{table_b['ci_hi']:.4f}] "
          f"p={table_b['wilcoxon_p']:.4f} delta={table_b['cliffs_delta']:.3f}")


if __name__ == "__main__":
    main()
