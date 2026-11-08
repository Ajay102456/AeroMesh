#!/usr/bin/env python3
"""Targeted re-run of the 17 campaign cells that got zero successful seeds
out of 162, all concentrated in the 'id' and 'vn' scenarios (larger
areaSide), which trigger the known ns-3 YansWifiPhy state-machine crash
under marginal-SNR broadcast collisions far more than 'ph' does. For each
missing cell, try many more perturbed seeds per seed-slot than the main
campaign did (up to MAX_TRIES_PER_SLOT), to get one valid run per slot.
"""
import csv
import os
import subprocess
import time

BIN = "./aeromesh_sim"
RESULTS_DIR = "results/campaign"
SIM_TIME = 90
AREA_BASE = 1000.0
PER_RUN_TIMEOUT = 420
MAX_TRIES_PER_SLOT = 15

MISSING_CELLS = [
    ("aodv", "id", 0, 16), ("aodv", "id", 0, 32), ("aodv", "id", 1, 16),
    ("aodv", "id", 2, 32), ("aodv", "vn", 0, 16), ("aodv", "vn", 0, 32),
    ("aodv", "vn", 1, 32), ("aodv", "vn", 2, 32),
    ("olsr", "id", 0, 32), ("olsr", "vn", 1, 32), ("olsr", "vn", 2, 16),
    ("olsr", "vn", 2, 32),
    ("aeromesh", "id", 0, 16), ("aeromesh", "id", 1, 16),
    ("aeromesh", "id", 2, 16), ("aeromesh", "vn", 1, 32),
    ("aeromesh", "vn", 2, 32),
]
SEED_SLOTS = [1, 2, 3]

log_path = os.path.join(RESULTS_DIR, "rerun_missing_log.csv")
log_rows = []
t0 = time.time()
total = len(MISSING_CELLS) * len(SEED_SLOTS)
done = 0

for mode, scenario, failLevel, nUav in MISSING_CELLS:
    for seed_slot in SEED_SLOTS:
        done += 1
        tag = f"{mode}_{scenario}_f{failLevel}_n{nUav}_s{seed_slot}"
        out_csv = os.path.join(RESULTS_DIR, f"{tag}.csv")
        ok = False
        tries = 0
        crashed = 0
        while tries < MAX_TRIES_PER_SLOT and not ok:
            use_seed = seed_slot + tries * 4007 + 1  # decorrelated offsets from the original campaign
            tries += 1
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
                    crashed += 1
                    if os.path.exists(out_csv):
                        os.remove(out_csv)
            except subprocess.TimeoutExpired:
                crashed += 1
                if os.path.exists(out_csv):
                    os.remove(out_csv)
        log_rows.append({
            "tag": tag, "mode": mode, "scenario": scenario, "failLevel": failLevel,
            "nUav": nUav, "seed_slot": seed_slot, "tries": tries,
            "crashed": crashed, "ok": ok,
        })
        elapsed = time.time() - t0
        print(f"[{done}/{total}] {tag} ok={ok} tries={tries} elapsed={elapsed:.0f}s", flush=True)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

n_failed = sum(1 for r in log_rows if not r["ok"])
print(f"\nDone. {total - n_failed}/{total} slots recovered. {n_failed} still unrecovered after {MAX_TRIES_PER_SLOT} tries each.")
