#!/usr/bin/env python3
"""Aggregate all per-run FlowMonitor CSVs in results/campaign/ into per-cell
and per-(mode,scenario,failLevel) summary statistics for the manuscript
Results section. Real ns-3 measured values only; cells with zero surviving
seeds are reported as N/A rather than estimated or interpolated.
"""
import csv
import glob
import os
from collections import defaultdict
from statistics import mean, pstdev

CLASS_NAMES = {5000: "responder", 5001: "medical", 5002: "civilian"}

# cell -> list of per-seed dicts: {class: {tx, rx, delaySum>0 count, ...}}
cells = defaultdict(list)

for path in glob.glob("results/campaign/*.csv"):
    base = os.path.basename(path)
    if base in ("campaign_log.csv", "rerun_missing_log.csv"):
        continue
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    mode = rows[0]["mode"]
    scenario = rows[0]["scenario"]
    failLevel = rows[0]["failLevel"]
    nUav = rows[0]["nUav"]
    seed = rows[0]["seed"]
    key = (mode, scenario, failLevel, nUav)

    per_class = {}
    for port, name in CLASS_NAMES.items():
        sub = [r for r in rows if int(r["dstPort"]) == port]
        tx = sum(int(r["txPackets"]) for r in sub)
        rx = sum(int(r["rxPackets"]) for r in sub)
        delays = [float(r["meanDelayMs"]) for r in sub if float(r["meanDelayMs"]) > 0]
        thr = sum(float(r["throughputKbps"]) for r in sub)
        per_class[name] = {
            "tx": tx, "rx": rx,
            "pdr": rx / tx if tx else 0.0,
            "meanDelayMs": mean(delays) if delays else 0.0,
            "throughputKbps": thr,
        }
    cells[key].append({"seed": seed, "classes": per_class})

# Write per-cell summary (mean +/- stdev across available seeds)
out_rows = []
for key, runs in sorted(cells.items()):
    mode, scenario, failLevel, nUav = key
    n = len(runs)
    row = {"mode": mode, "scenario": scenario, "failLevel": failLevel,
           "nUav": nUav, "n_seeds": n}
    for cname in CLASS_NAMES.values():
        pdrs = [r["classes"][cname]["pdr"] for r in runs]
        delays = [r["classes"][cname]["meanDelayMs"] for r in runs if r["classes"][cname]["meanDelayMs"] > 0]
        thrs = [r["classes"][cname]["throughputKbps"] for r in runs]
        row[f"{cname}_pdr_mean"] = mean(pdrs) if pdrs else float("nan")
        row[f"{cname}_pdr_std"] = pstdev(pdrs) if len(pdrs) > 1 else 0.0
        row[f"{cname}_delay_mean_ms"] = mean(delays) if delays else float("nan")
        row[f"{cname}_throughput_kbps_mean"] = mean(thrs) if thrs else float("nan")
    out_rows.append(row)

os.makedirs("results/summary", exist_ok=True)
with open("results/summary/cell_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"{len(out_rows)} cells summarized -> results/summary/cell_summary.csv")

# Also a coarser mode x scenario rollup (averaged across failLevel/nUav/seed) for headline numbers
mode_scenario = defaultdict(lambda: defaultdict(list))
for key, runs in cells.items():
    mode, scenario, failLevel, nUav = key
    for r in runs:
        for cname in CLASS_NAMES.values():
            mode_scenario[(mode, scenario)][f"{cname}_pdr"].append(r["classes"][cname]["pdr"])
            if r["classes"][cname]["meanDelayMs"] > 0:
                mode_scenario[(mode, scenario)][f"{cname}_delay"].append(r["classes"][cname]["meanDelayMs"])

with open("results/summary/mode_scenario_summary.csv", "w", newline="") as f:
    fieldnames = ["mode", "scenario", "n_runs",
                  "responder_pdr_mean", "medical_pdr_mean", "civilian_pdr_mean",
                  "responder_delay_mean_ms", "medical_delay_mean_ms", "civilian_delay_mean_ms"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for (mode, scenario), d in sorted(mode_scenario.items()):
        n_runs = len(d["responder_pdr"])
        w.writerow({
            "mode": mode, "scenario": scenario, "n_runs": n_runs,
            "responder_pdr_mean": mean(d["responder_pdr"]) if d["responder_pdr"] else float("nan"),
            "medical_pdr_mean": mean(d["medical_pdr"]) if d["medical_pdr"] else float("nan"),
            "civilian_pdr_mean": mean(d["civilian_pdr"]) if d["civilian_pdr"] else float("nan"),
            "responder_delay_mean_ms": mean(d["responder_delay"]) if d["responder_delay"] else float("nan"),
            "medical_delay_mean_ms": mean(d["medical_delay"]) if d["medical_delay"] else float("nan"),
            "civilian_delay_mean_ms": mean(d["civilian_delay"]) if d["civilian_delay"] else float("nan"),
        })

print("mode x scenario rollup -> results/summary/mode_scenario_summary.csv")

missing = [k for k, v in cells.items() if len(v) == 0]
all_keys = set()
MODES = ["aodv", "olsr", "aeromesh"]
SCENARIOS = ["ph", "id", "vn"]
FAIL_LEVELS = ["0", "1", "2"]
FLEET_SIZES = ["16", "32"]
for m in MODES:
    for s in SCENARIOS:
        for f in FAIL_LEVELS:
            for n in FLEET_SIZES:
                all_keys.add((m, s, f, n))
zero_seed_cells = [k for k in all_keys if len(cells.get(k, [])) == 0]
print(f"\nCells with zero surviving seeds (of {len(all_keys)} total cells): {len(zero_seed_cells)}")
for k in sorted(zero_seed_cells):
    print("  ", k)
