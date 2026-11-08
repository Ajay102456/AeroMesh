"""Joint network+inference co-simulator.

Implements the auction/deferral control logic of Algorithm 2 (Section
methods) over a 10-epoch, 300-second mission horizon matching the real ns-3
campaign's simTime=300 default. Connectivity Phi(t) is drawn by bootstrap
resampling from the real per-seed ns-3 FlowMonitor records; assessment value
Psi(t) is drawn from the real per-severity accuracy of the trained
early-exit classifier. The epoch-to-severity recovery schedule and the
auction's relay/inference role split are declared simulation-protocol
parameters, not measured quantities; they are stated plainly here and in the
manuscript rather than disguised as additional ns-3 output.
"""
import random
from data import (CLASSES, FAIL_LEVELS, bootstrap_sample,
                   load_inference_curves, load_per_seed_pool,
                   relay_capacity_slope)

N_EPOCHS = 10
EPOCH_S = 30.0
N_UAV = 16
N_INFERENCE_BASE = 5          # round(16 * 1/3), the 2:1 relay:inference ratio
MAX_EXTRA_RELAY = 3           # auction's declared relay-promotion cap
RECOVERY_EVERY_N_EPOCHS = 4   # declared self-healing recovery dwell time

W_PRIORITY = {"responder": 10.0, "medical": 3.0, "civilian": 1.0}
W_UNIFORM = {"responder": 1.0, "medical": 1.0, "civilian": 1.0}

# epoch severity -> corruption severity bucket fed to the inference layer
SEVERITY_TO_CORRUPTION = {0: 0, 1: 2, 2: 4}

ABLATIONS = [
    "full",              # AeroMesh: self-healing + auction + early-exit + conformal
    "no_self_healing",   # (1) topology loop disabled -> real AODV data substituted
    "no_priority",       # (2) uniform class weights
    "no_early_exit",     # (3) vanilla (non-cascaded) softmax accuracy curve
    "no_uncertainty",     # (4) cascade without conformal calibration
    "fixed_roles",       # (5) no auction: static relay/inference split
    "oracle",            # (6) idealized global controller upper bound
]


def severity_schedule(start_severity, self_healing):
    sched = []
    s = start_severity
    for epoch in range(N_EPOCHS):
        if self_healing and epoch > 0 and epoch % RECOVERY_EVERY_N_EPOCHS == 0:
            s = max(s - 1, 0)
        sched.append(s)
    return sched


def run_one(pool, curves, scenario, start_failLevel, ablation, rng):
    self_healing = (ablation != "no_self_healing")
    network_mode = "aodv" if ablation == "no_self_healing" else "aeromesh"
    weights = W_UNIFORM if ablation == "no_priority" else W_PRIORITY

    if ablation == "no_early_exit":
        acc_curve = curves["risk_coverage_vanilla"]
    elif ablation == "no_uncertainty":
        acc_curve = curves["risk_coverage_early_exit"]
    else:
        acc_curve = curves["risk_coverage_full_conformal"]

    slope = relay_capacity_slope(pool, network_mode, scenario, str(start_failLevel))
    sched = severity_schedule(start_failLevel, self_healing)

    # Draw one bootstrap realization per distinct severity level reached in
    # this run's schedule, held fixed across the epochs at that severity,
    # rather than re-rolling every epoch -- this represents one coherent
    # realized mission trace rather than an epoch-independent resample.
    draws = {}
    flagged_any = False
    for sev in set(sched):
        rec, flagged = bootstrap_sample(pool, network_mode, scenario, str(sev), "16", rng)
        draws[sev] = rec
        flagged_any = flagged_any or flagged

    phi_series, psi_series = [], []
    phi_class_series = {c: [] for c in CLASSES}
    for epoch, sev in enumerate(sched):
        rec = draws[sev]

        if ablation == "oracle":
            extra_relay = MAX_EXTRA_RELAY
            n_inf = N_INFERENCE_BASE
        elif ablation == "fixed_roles":
            extra_relay = 0
            n_inf = N_INFERENCE_BASE
        else:
            extra_relay = min(sev, MAX_EXTRA_RELAY)
            n_inf = max(N_INFERENCE_BASE - sev, 2)

        class_s = {}
        for c in CLASSES:
            base_pdr = rec.get(f"{c}_pdr")
            if base_pdr is None:
                base_pdr = 0.0
            class_s[c] = min(base_pdr + extra_relay * slope[c], 1.0)
            phi_class_series[c].append(class_s[c])

        num = sum(weights[c] * class_s[c] for c in CLASSES)
        den = sum(weights.values())
        phi_series.append(num / den)

        corruption = SEVERITY_TO_CORRUPTION[sev]
        acc = acc_curve.get(corruption, acc_curve[max(acc_curve)])
        psi_series.append(n_inf * acc)

    return {
        "phi": phi_series,
        "phi_class": phi_class_series,
        "psi": psi_series,
        "flagged_fallback": flagged_any,
    }


def time_to_restore(phi_class_series, target_class_pdr, tol=0.9):
    for epoch, v in enumerate(phi_class_series):
        if v >= tol * target_class_pdr:
            return (epoch + 1) * EPOCH_S
    return None  # not restored within the mission horizon


def main():
    pool = load_per_seed_pool()
    curves = load_inference_curves()
    recovered_target = {}
    for scenario in ["ph", "id", "vn"]:
        recovered_target[scenario] = {}
        for c in CLASSES:
            key = ("aeromesh", scenario, "0", "16")
            vals = [r[f"{c}_pdr"] for r in pool.get(key, []) if r.get(f"{c}_pdr") is not None]
            recovered_target[scenario][c] = sum(vals) / len(vals) if vals else 0.8

    import json as _json
    results = []
    for scenario in ["ph", "id", "vn"]:
        for failLevel in [0, 1, 2]:
            for ablation in ABLATIONS:
                for rep in range(5):
                    # Common-random-numbers: the RNG seed depends only on
                    # (scenario, failLevel, replicate), not on ablation, so
                    # every ablation in a replicate draws from the same
                    # underlying random stream. This is what makes the
                    # paired bootstrap comparison in Table B meaningful
                    # rather than comparing independent noise.
                    rng = random.Random(hash((scenario, failLevel, rep)) & 0xffffffff)
                    out = run_one(pool, curves, scenario, failLevel, ablation, rng)
                    ttr = {c: time_to_restore(out["phi_class"][c], recovered_target[scenario][c])
                           for c in CLASSES}
                    results.append({
                        "scenario": scenario, "failLevel": failLevel,
                        "ablation": ablation, "replicate": rep,
                        "phi_series": out["phi"], "psi_series": out["psi"],
                        "phi_mean": sum(out["phi"]) / len(out["phi"]),
                        "psi_mean": sum(out["psi"]) / len(out["psi"]),
                        "time_to_restore_s": ttr,
                        "used_fallback_cell": out["flagged_fallback"],
                    })
    with open("cosim_results.json", "w") as f:
        _json.dump(results, f, indent=1)
    print(f"wrote {len(results)} co-simulator runs to cosim_results.json")


if __name__ == "__main__":
    main()
