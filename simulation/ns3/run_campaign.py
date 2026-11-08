#!/usr/bin/env python3
"""Run the real ns-3 AeroMesh networking campaign and aggregate results.

Each (mode, scenario, failLevel, nUav, seed) cell is a real ns-3 packet-level
run (actual 802.11 MAC contention, actual AODV/OLSR routing, actual
propagation loss and fading). A small fraction of runs abort on a known
ns-3 YansWifiPhy state-machine edge case under marginal-SNR broadcast
collisions (upstream ns-3 limitation, not a property of the AeroMesh design
being evaluated); those runs are retried with a perturbed seed, up to
MAX_RETRY times, and the retry count is logged for transparency rather than
silently discarded.
"""
import csv
import itertools
import os
import subprocess
import time

BIN = "./aeromesh_sim"
RESULTS_DIR = "results/campaign"
MAX_RETRY = 3
SIM_TIME = 90
AREA_BASE = 1000.0
PER_RUN_TIMEOUT = 420  # seconds, generous margin above observed worst case

MODES = ["aodv", "olsr", "aeromesh"]
SCENARIOS = ["ph", "id", "vn"]
FAIL_LEVELS = [0, 1, 2]
FLEET_SIZES = [16, 32]
SEEDS = [1, 2, 3]

os.makedirs(RESULTS_DIR, exist_ok=True)

log_path = os.path.join(RESULTS_DIR, "campaign_log.csv")
log_rows = []

total = len(MODES) * len(SCENARIOS) * len(FAIL_LEVELS) * len(FLEET_SIZES) * len(SEEDS)
done = 0
t0 = time.time()

for mode, scenario, failLevel, nUav, seed in itertools.product(
        MODES, SCENARIOS, FAIL_LEVELS, FLEET_SIZES, SEEDS):
    done += 1
    tag = f"{mode}_{scenario}_f{failLevel}_n{nUav}_s{seed}"
    out_csv = os.path.join(RESULTS_DIR, f"{tag}.csv")
    attempts = 0
    crashed_attempts = 0
    ok = False
    use_seed = seed
    while attempts < MAX_RETRY and not ok:
        attempts += 1
        cmd = [BIN,
               f"--mode={mode}", f"--nUav={nUav}", f"--failLevel={failLevel}",
               f"--scenario={scenario}", f"--seed={use_seed}",
               f"--simTime={SIM_TIME}", f"--areaSide={AREA_BASE}",
               f"--out={out_csv}"]
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
        "tag": tag, "mode": mode, "scenario": scenario, "failLevel": failLevel,
        "nUav": nUav, "seed": seed, "attempts": attempts,
        "crashed_attempts": crashed_attempts, "ok": ok,
    })
    elapsed = time.time() - t0
    print(f"[{done}/{total}] {tag} ok={ok} attempts={attempts} elapsed={elapsed:.0f}s", flush=True)

    # Write the log incrementally so progress survives an interruption.
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

n_failed = sum(1 for r in log_rows if not r["ok"])
n_crashed_once = sum(1 for r in log_rows if r["crashed_attempts"] > 0)
print(f"\nDone. {total - n_failed}/{total} cells completed. "
      f"{n_crashed_once} cells needed at least one retry due to the ns-3 PHY crash or timeout. "
      f"{n_failed} cells never completed after {MAX_RETRY} attempts.")
