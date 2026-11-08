#!/usr/bin/env python3
"""Turn cached evaluate.py outputs (.npz of per-exit softmax probabilities
over real xBD/Palu crops, clean and under the corruption pipeline) into
the manuscript's Figure C (risk-coverage curves), Figure D (cross-dataset
transfer), and a conformal-coverage table. All numbers here are computed
directly from measured model outputs; nothing is interpolated or assumed.
"""
import argparse
import json
import numpy as np

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def risk_coverage_curve(probs, labels, coverage_levels):
    """Confidence-based selective classification: accept the top-coverage
    fraction by max-softmax confidence, report error rate (risk) among
    accepted examples, following Geifman & El-Yaniv's selective risk."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels)
    order = np.argsort(-conf)
    n = len(labels)
    out = []
    for phi in coverage_levels:
        k = max(1, int(round(phi * n)))
        idx = order[:k]
        risk = 1.0 - correct[idx].mean()
        out.append(risk)
    return out


def conformal_threshold(calib_probs, calib_labels, alpha):
    """Standard inductive (split) conformal calibration. Nonconformity
    score = 1 - softmax probability assigned to the true label. Returns
    the (1-alpha) empirical quantile threshold q_hat."""
    n = len(calib_labels)
    scores = 1.0 - calib_probs[np.arange(n), calib_labels]
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = np.quantile(scores, q_level)
    return q_hat


def conformal_eval(test_probs, test_labels, q_hat):
    """Prediction set = {classes with softmax >= 1 - q_hat}. Reports
    empirical coverage (true label in set) and mean set size, and the
    risk/coverage pair when restricting to confident singleton sets."""
    sets = test_probs >= (1.0 - q_hat)
    sets[sets.sum(axis=1) == 0] = True  # guard against empty sets
    n = len(test_labels)
    covered = sets[np.arange(n), test_labels].mean()
    mean_size = sets.sum(axis=1).mean()
    singleton = sets.sum(axis=1) == 1
    if singleton.sum() > 0:
        singleton_pred = test_probs[singleton].argmax(axis=1)
        singleton_acc = (singleton_pred == test_labels[singleton]).mean()
        singleton_coverage_frac = singleton.mean()
    else:
        singleton_acc, singleton_coverage_frac = float("nan"), 0.0
    return {
        "empirical_coverage": float(covered),
        "mean_set_size": float(mean_size),
        "singleton_accept_fraction": float(singleton_coverage_frac),
        "singleton_risk": float(1 - singleton_acc) if not np.isnan(singleton_acc) else None,
    }


def early_exit_cascade_eval(probs1, probs2, probs3, labels, entropy_thresh=(0.3, 0.5)):
    """BranchyNet-style cascade: accept exit1 if its normalized entropy is
    below entropy_thresh[0]; else try exit2 with entropy_thresh[1]; else
    fall through to exit3 (always accepted). Reports accuracy and the
    fraction of tiles resolved at each depth (a real proxy for the
    compute savings early exiting is supposed to buy)."""
    def norm_entropy(p):
        p = np.clip(p, 1e-9, 1.0)
        h = -(p * np.log(p)).sum(axis=1)
        return h / np.log(p.shape[1])

    h1 = norm_entropy(probs1)
    h2 = norm_entropy(probs2)
    exit_at = np.full(len(labels), 3)
    exit_at[h1 < entropy_thresh[0]] = 1
    mask2 = (exit_at == 3) & (h2 < entropy_thresh[1])
    exit_at[mask2] = 2

    preds = np.where(exit_at[:, None] == 1, probs1.argmax(1)[:, None],
             np.where(exit_at[:, None] == 2, probs2.argmax(1)[:, None], probs3.argmax(1)[:, None]))[:, 0]
    acc = (preds == labels).mean()
    frac = {d: float((exit_at == d).mean()) for d in [1, 2, 3]}
    return {"accuracy": float(acc), "exit_fraction": frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True, help="evaluate.py output on the in-domain test set (also used for clean risk-coverage)")
    ap.add_argument("--calib_npz", required=True, help="evaluate.py output on a held-out calibration set (clean only needed)")
    ap.add_argument("--domain_name", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    test = np.load(args.test_npz)
    calib = np.load(args.calib_npz)
    labels = test["labels"]
    calib_labels = calib["labels"]

    coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    severities = [0, 1, 2, 3, 4, 5]
    corruptions = ["fog", "smoke", "low_light", "motion_blur", "tile_loss"]

    report = {"domain": args.domain_name, "n_test": int(len(labels)), "n_calib": int(len(calib_labels))}

    # Figure C data: risk-coverage curves at each corruption severity (averaged across corruption types), three configs
    rc_vanilla = {}
    rc_earlyexit = {}
    rc_conformal = {}
    for sev in severities:
        risks_vanilla, risks_ee, risks_conf = [], [], []
        ctypes = ["clean"] if sev == 0 else corruptions
        for ctype in ctypes:
            key = f"{ctype}_s{sev}" if sev > 0 else "clean_s0"
            if f"{key}_exit3" not in test:
                continue
            p1, p2, p3 = test[f"{key}_exit1"], test[f"{key}_exit2"], test[f"{key}_exit3"]
            risks_vanilla.append(risk_coverage_curve(p3, labels, coverage_levels))

            cascade = early_exit_cascade_eval(p1, p2, p3, labels)
            risks_ee.append([1 - cascade["accuracy"]] * len(coverage_levels))

            if f"{key}_exit3" in calib:
                cp3 = calib[f"{key}_exit3"]
            else:
                cp3 = calib["clean_s0_exit3"]
            conf_risks = []
            for phi in coverage_levels:
                alpha = max(0.01, 1 - phi)
                q_hat = conformal_threshold(cp3, calib_labels, alpha)
                res = conformal_eval(p3, labels, q_hat)
                conf_risks.append(res["singleton_risk"] if res["singleton_risk"] is not None else 1.0)
            risks_conf.append(conf_risks)
        if risks_vanilla:
            rc_vanilla[sev] = np.mean(risks_vanilla, axis=0).tolist()
            rc_earlyexit[sev] = np.mean(risks_ee, axis=0).tolist()
            rc_conformal[sev] = np.mean(risks_conf, axis=0).tolist()

    report["coverage_levels"] = coverage_levels
    report["risk_coverage_vanilla"] = rc_vanilla
    report["risk_coverage_early_exit"] = rc_earlyexit
    report["risk_coverage_full_conformal"] = rc_conformal

    # clean accuracy per exit + cascade exit-depth distribution (clean)
    p1, p2, p3 = test["clean_s0_exit1"], test["clean_s0_exit2"], test["clean_s0_exit3"]
    report["clean_accuracy_exit1"] = float((p1.argmax(1) == labels).mean())
    report["clean_accuracy_exit2"] = float((p2.argmax(1) == labels).mean())
    report["clean_accuracy_exit3"] = float((p3.argmax(1) == labels).mean())
    report["clean_cascade"] = early_exit_cascade_eval(p1, p2, p3, labels)

    # conformal coverage at alpha=0.1 on clean data, standard headline number
    q_hat_90 = conformal_threshold(calib["clean_s0_exit3"], calib_labels, alpha=0.1)
    report["conformal_clean_alpha0.1"] = conformal_eval(p3, labels, q_hat_90)

    # per-class accuracy (final exit, clean) for the confusion breakdown
    per_class = {}
    for i, c in enumerate(CLASSES):
        mask = labels == i
        if mask.sum() > 0:
            per_class[c] = {"n": int(mask.sum()), "acc": float((p3.argmax(1)[mask] == i).mean())}
        else:
            per_class[c] = {"n": 0, "acc": None}
    report["per_class_clean_accuracy"] = per_class

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {args.out_json}")
    print(json.dumps({k: v for k, v in report.items() if k not in
                       ("risk_coverage_vanilla", "risk_coverage_early_exit", "risk_coverage_full_conformal")}, indent=2))


if __name__ == "__main__":
    main()
