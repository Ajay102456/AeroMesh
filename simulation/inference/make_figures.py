#!/usr/bin/env python3
"""Generate Figure C (risk-coverage curves) and Figure D (cross-dataset
transfer) for the manuscript from the real analysis_*.json outputs
produced by analyze_results.py. Saves both PDF and SVG, matching the
convention already used for figures/architecture.* and figures/scenarios.*.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/s/Documents/AeroMesh/manuscript/figures"


def load(name):
    with open(f"/home/s/Documents/AeroMesh/simulation/inference/{name}") as f:
        return json.load(f)


def figure_c():
    indomain = load("analysis_nonpalu_indomain.json")
    transfer = load("analysis_nonpalu_to_palu.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, report, title in [(axes[0], indomain, "In-domain (non-Palu test)"),
                               (axes[1], transfer, "Cross-domain transfer (Palu test)")]:
        cov = report["coverage_levels"]
        for key, label, style in [
            ("risk_coverage_vanilla", "Vanilla softmax", "o-"),
            ("risk_coverage_early_exit", "Early-exit cascade", "s-"),
            ("risk_coverage_full_conformal", "Early-exit + conformal", "^-"),
        ]:
            sev0 = report[key].get("0")
            sev3 = report[key].get("3")
            if sev0:
                ax.plot(cov, sev0, style, label=f"{label} (clean)", alpha=0.9)
            if sev3:
                ax.plot(cov, sev3, style, label=f"{label} (severity 3)", alpha=0.5, linestyle="--")
        ax.set_xlabel("Coverage $\\phi$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Selective risk (error rate on accepted tiles)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Figure C. Risk-coverage curves, clean vs. corruption severity 3", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/risk_coverage.pdf")
    fig.savefig(f"{OUT_DIR}/risk_coverage.svg")
    print("wrote risk_coverage.pdf/svg")


def figure_d():
    nonpalu_in = load("analysis_nonpalu_indomain.json")
    nonpalu_to_palu = load("analysis_nonpalu_to_palu.json")
    palu_in = load("analysis_palu_indomain.json")
    palu_to_nonpalu = load("analysis_palu_to_nonpalu.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    labels = ["Non-Palu\nin-domain", "Trained non-Palu\n-> Palu test", "Palu\nin-domain", "Trained Palu\n-> non-Palu test"]
    accs = [nonpalu_in["clean_cascade"]["accuracy"], nonpalu_to_palu["clean_cascade"]["accuracy"],
            palu_in["clean_cascade"]["accuracy"], palu_to_nonpalu["clean_cascade"]["accuracy"]]
    colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]
    axes[0].bar(labels, accs, color=colors)
    axes[0].set_ylabel("Cascade accuracy (clean imagery)")
    axes[0].set_title("Cross-dataset transfer accuracy drop", fontsize=10)
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].tick_params(axis="x", labelsize=7)

    cov_labels = ["Non-Palu\nin-domain", "Non-Palu model\n-> Palu test", "Palu\nin-domain", "Palu model\n-> non-Palu test"]
    coverage = [nonpalu_in["conformal_clean_alpha0.1"]["empirical_coverage"],
                nonpalu_to_palu["conformal_clean_alpha0.1"]["empirical_coverage"],
                palu_in["conformal_clean_alpha0.1"]["empirical_coverage"],
                palu_to_nonpalu["conformal_clean_alpha0.1"]["empirical_coverage"]]
    singleton_risk = [nonpalu_in["conformal_clean_alpha0.1"]["singleton_risk"],
                       nonpalu_to_palu["conformal_clean_alpha0.1"]["singleton_risk"],
                       palu_in["conformal_clean_alpha0.1"]["singleton_risk"],
                       palu_to_nonpalu["conformal_clean_alpha0.1"]["singleton_risk"]]
    x = range(len(cov_labels))
    axes[1].bar([i - 0.2 for i in x], coverage, width=0.4, label="Empirical set coverage (target 0.90)", color="#55A868")
    axes[1].bar([i + 0.2 for i in x], singleton_risk, width=0.4, label="Risk among confident (singleton) predictions", color="#C44E52")
    axes[1].axhline(0.90, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(cov_labels, fontsize=7)
    axes[1].set_title("Conformal coverage vs. confident-prediction risk", fontsize=10)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_ylim(0, 1.05)

    fig.suptitle("Figure D. Cross-dataset transfer: accuracy and conformal behavior", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cross_dataset_transfer.pdf")
    fig.savefig(f"{OUT_DIR}/cross_dataset_transfer.svg")
    print("wrote cross_dataset_transfer.pdf/svg")


if __name__ == "__main__":
    figure_c()
    figure_d()
