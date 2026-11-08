#!/usr/bin/env python3
"""Run the real joint network+role-auction ns-3 campaign.

Each (ablation, scenario, startFailLevel, seed) cell is a real ns-3
packet-level run of aeromesh_joint_sim, which genuinely withdraws UAVs from
relay duty for inference windows (see aeromesh_joint_sim.cc) and snapshots
FlowMonitor at every epoch boundary for a real time-resolved per-class PDR
series. Three ablations are covered, matching the two real couplings most
load-bearing for the manuscript's Figure A and Table B claims:

  full            - selfHealing=1, rolePolicy=auction  (AeroMesh)
  no_self_healing - selfHealing=0, rolePolicy=auction  (no repositioning)
  fixed_roles     - selfHealing=1, rolePolicy=fixed    (no auction)

A small fraction of cells abort on the same known ns-3 YansWifiPhy
state-machine edge case already documented in run_campaign.py; those are
retried with a perturbed seed up to MAX_RETRY times, and any cell that
never completes is logged as such rather than silently omitted.
"""
import csv
import itertools
import os
import subprocess
import time

BIN = "./aeromesh_joint_sim"
RESULTS_DIR = "results/joint"
MAX_RETRY = 3
EPOCH_LEN = 30.0
N_EPOCHS = 10
PER_RUN_TIMEOUT = 300

ABLATIONS = {
    "full": {"selfHealing": 1, "rolePolicy": "auction"},
    "no_self_healing": {"selfHealing": 0, "rolePolicy": "auction"},
    "fixed_roles": {"selfHealing": 1, "rolePolicy": "fixed"},
}
SCENARIOS = ["ph"]
# id and vn are excluded: the 300s joint mission horizon runs into ns-3's
# known YansWifiPhy marginal-SNR state-machine limitation at ~100% failure
# in their wider simulated area, even with the full retry budget below (see
# the manuscript's Limitations section for the measured ph-only reliability
# finding). ph is unaffected and run here as real joint ns-3 data; id/vn
# continue to use the reduced-fidelity Python co-simulator.
FAIL_LEVELS = [0, 1, 2]
SEEDS = [1, 2, 3]

os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = os.path.join(RESULTS_DIR, "joint_campaign_log.csv")
log_rows = []

total = len(ABLATIONS) * len(SCENARIOS) * len(FAIL_LEVELS) * len(SEEDS)
done = 0
t0 = time.time()

for ablation, scenario, failLevel, seed in itertools.product(
        ABLATIONS, SCENARIOS, FAIL_LEVELS, SEEDS):
    done += 1
    params = ABLATIONS[ablation]
    tag = f"{ablation}_{scenario}_f{failLevel}_s{seed}"
    out_csv = os.path.join(RESULTS_DIR, f"{tag}.csv")
    attempts = 0
    crashed_attempts = 0
    ok = False
    use_seed = seed
    while attempts < MAX_RETRY and not ok:
        attempts += 1
        cmd = [BIN,
               f"--scenario={scenario}", f"--startFailLevel={failLevel}",
               f"--selfHealing={params['selfHealing']}",
               f"--rolePolicy={params['rolePolicy']}",
               f"--seed={use_seed}", f"--epochLen={EPOCH_LEN}", f"--nEpochs={N_EPOCHS}",
               f"--snapOut={out_csv}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_RUN_TIMEOUT)
            if r.returncode == 0 and os.path.exists(out_csv):
                ok = True
            else:
                crashed_attempts += 1
                use_seed = seed + attempts * 1000
                if os.path.exists(out_csv):
                    os.remove(out_csv)
        except subprocess.TimeoutExpired:
            crashed_attempts += 1
            use_seed = seed + attempts * 1000
            if os.path.exists(out_csv):
                os.remove(out_csv)
    log_rows.append({
        "tag": tag, "ablation": ablation, "scenario": scenario, "failLevel": failLevel,
        "seed": seed, "attempts": attempts, "crashed_attempts": crashed_attempts, "ok": ok,
    })
    elapsed = time.time() - t0
    print(f"[{done}/{total}] {tag} ok={ok} attempts={attempts} elapsed={elapsed:.0f}s", flush=True)

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

n_failed = sum(1 for r in log_rows if not r["ok"])
n_crashed_once = sum(1 for r in log_rows if r["crashed_attempts"] > 0)
print(f"\nDone. {total - n_failed}/{total} cells completed. "
      f"{n_crashed_once} cells needed at least one retry. "
      f"{n_failed} cells never completed after {MAX_RETRY} attempts.")
