"""Statistical analysis of the co-simulator output: Table A (time-to-restore),
Table B (six-ablation paired comparison against full AeroMesh, bootstrap CI,
Wilcoxon signed-rank, Cliff's delta, Holm-Bonferroni correction), and the
Figure A / Figure B source arrays, exactly as specified in Section
experiments of the manuscript.
"""
import json
from collections import defaultdict

import numpy as np
from scipy import stats

ABLATIONS_NON_FULL = ["no_self_healing", "no_priority", "no_early_exit",
                       "no_uncertainty", "fixed_roles", "oracle"]
SCENARIOS = ["ph", "id", "vn"]
CLASSES = ["responder", "medical", "civilian"]


def composite_score(run, psi_norm):
    """Mean of Phi and Psi normalized to full AeroMesh's own Psi for the
    same (scenario, failLevel), so the two terms of the composite are on a
    comparable scale rather than Psi's raw tile-count units dominating."""
    return 0.5 * run["phi_mean"] + 0.5 * (run["psi_mean"] / psi_norm)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def paired_bootstrap_ci(diffs, n_resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    n = len(diffs)
    boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean()
                            for _ in range(n_resamples)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diffs.mean(), lo, hi


def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = pvals[idx] * (m - rank)
        running_max = max(running_max, corrected)
        adj[idx] = min(running_max, 1.0)
    return adj


def main():
    results = json.load(open("cosim_results.json"))
    by_key = defaultdict(dict)
    for r in results:
        by_key[(r["scenario"], r["failLevel"], r["replicate"])][r["ablation"]] = r

    # Normalize Psi within each (scenario, failLevel) by full AeroMesh's own
    # mean Psi across replicates, so the composite score's two halves are
    # comparable in magnitude.
    psi_norm = {}
    for scenario in SCENARIOS:
        for fl in (0, 1, 2):
            vals = [r["psi_mean"] for r in results
                     if r["scenario"] == scenario and r["failLevel"] == fl and r["ablation"] == "full"]
            psi_norm[(scenario, fl)] = np.mean(vals)

    # ---- Table B: full vs each ablation, paired by (scenario, failLevel, replicate) ----
    table_b = []
    full_scores_by_pair = {}
    for key, runs in by_key.items():
        scenario, fl, rep = key
        full_scores_by_pair[key] = composite_score(runs["full"], psi_norm[(scenario, fl)])

    pvals_raw = []
    rows_tmp = []
    for ablation in ABLATIONS_NON_FULL:
        full_vals, abl_vals = [], []
        for key, runs in by_key.items():
            scenario, fl, rep = key
            if ablation not in runs:
                continue
            full_vals.append(full_scores_by_pair[key])
            abl_vals.append(composite_score(runs[ablation], psi_norm[(scenario, fl)]))
        diffs = np.array(full_vals) - np.array(abl_vals)  # positive = AeroMesh full wins
        mean_diff, lo, hi = paired_bootstrap_ci(diffs)
        try:
            wstat, pval = stats.wilcoxon(full_vals, abl_vals)
        except ValueError:
            pval = 1.0
        delta = cliffs_delta(full_vals, abl_vals)
        pvals_raw.append(pval)
        rows_tmp.append({
            "ablation": ablation, "n_pairs": len(diffs),
            "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
            "wilcoxon_p_raw": pval, "cliffs_delta": delta,
        })
    adj = holm_bonferroni(np.array(pvals_raw))
    for row, p_adj in zip(rows_tmp, adj):
        row["wilcoxon_p_holm"] = float(p_adj)
        table_b.append(row)

    # ---- Table A: time-to-restore per priority class, full AeroMesh vs no_self_healing ----
    table_a = []
    for scenario in SCENARIOS:
        for fl in (0, 1, 2):
            for ablation in ["full", "no_self_healing"]:
                for c in CLASSES:
                    vals = [r["time_to_restore_s"][c] for r in results
                            if r["scenario"] == scenario and r["failLevel"] == fl and r["ablation"] == ablation]
                    n_restored = sum(1 for v in vals if v is not None)
                    restored_vals = [v for v in vals if v is not None]
                    table_a.append({
                        "scenario": scenario, "failLevel": fl, "ablation": ablation,
                        "priority_class": c, "n_runs": len(vals), "n_restored": n_restored,
                        "mean_restore_s": float(np.mean(restored_vals)) if restored_vals else None,
                    })

    # ---- Figure A source: mean Phi(t) trajectory per scenario/failLevel, full vs no_self_healing ----
    fig_a = []
    for scenario in SCENARIOS:
        for fl in (0, 1, 2):
            for ablation in ["full", "no_self_healing"]:
                series = [r["phi_series"] for r in results
                          if r["scenario"] == scenario and r["failLevel"] == fl and r["ablation"] == ablation]
                mean_series = np.mean(series, axis=0).tolist()
                fig_a.append({"scenario": scenario, "failLevel": fl, "ablation": ablation,
                               "phi_t": mean_series})

    # ---- Figure B source: joint (Phi, Psi) scatter across all configs/replicates ----
    fig_b = [{"scenario": r["scenario"], "failLevel": r["failLevel"], "ablation": r["ablation"],
              "phi_mean": r["phi_mean"], "psi_mean": r["psi_mean"],
              "psi_norm": r["psi_mean"] / psi_norm[(r["scenario"], r["failLevel"])]}
             for r in results]

    json.dump(table_a, open("table_a.json", "w"), indent=1)
    json.dump(table_b, open("table_b.json", "w"), indent=1)
    json.dump(fig_a, open("fig_a_data.json", "w"), indent=1)
    json.dump(fig_b, open("fig_b_data.json", "w"), indent=1)
    print("Table A rows:", len(table_a))
    print("Table B rows:", len(table_b))
    for row in table_b:
        print(row["ablation"], f"diff={row['mean_diff']:.4f}",
              f"CI=[{row['ci_lo']:.4f},{row['ci_hi']:.4f}]",
              f"p_holm={row['wilcoxon_p_holm']:.4f}",
              f"delta={row['cliffs_delta']:.3f}")


if __name__ == "__main__":
    main()
