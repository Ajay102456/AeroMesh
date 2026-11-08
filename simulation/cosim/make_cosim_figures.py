#!/usr/bin/env python3
"""Generate Figure A (time-resolved Phi(t)) and Figure B (joint Pareto
frontier) from the co-simulator's real output, matching the manuscript's
existing figures/*.pdf + .svg convention.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/home/s/Documents/AeroMesh/manuscript/figures"
EPOCH_S = 30.0


def figure_a():
    data = json.load(open("fig_a_data.json"))
    scenarios = ["ph", "id", "vn"]
    names = {"ph": "Philippines (urban typhoon)", "id": "Indonesia (archipelagic earthquake)",
             "vn": "Vietnam (gradual monsoon flood)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    t = [(_ + 1) * EPOCH_S for _ in range(10)]
    for ax, scenario in zip(axes, scenarios):
        for fl, color in zip((0, 1, 2), ("#4C72B0", "#DD8452", "#C44E52")):
            full = next(d for d in data if d["scenario"] == scenario and d["failLevel"] == fl and d["ablation"] == "full")
            nosh = next(d for d in data if d["scenario"] == scenario and d["failLevel"] == fl and d["ablation"] == "no_self_healing")
            ax.plot(t, full["phi_t"], "-", color=color, label=f"AeroMesh, failLevel {fl}")
            ax.plot(t, nosh["phi_t"], "--", color=color, alpha=0.6, label=f"No self-healing, failLevel {fl}")
        ax.set_title(names[scenario], fontsize=9)
        ax.set_xlabel("Mission time (s)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Priority-weighted connectivity $\Phi(t)$")
    axes[-1].legend(fontsize=6, loc="lower right", ncol=1)
    fig.suptitle("Figure A. Co-simulator connectivity-restoration trajectory, AeroMesh vs. no-self-healing ablation", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cosim_phi_t.pdf")
    fig.savefig(f"{OUT_DIR}/cosim_phi_t.svg")
    print("wrote cosim_phi_t.pdf/svg")


def figure_b():
    data = json.load(open("fig_b_data.json"))
    ablation_colors = {
        "full": "#4C72B0", "no_self_healing": "#DD8452", "no_priority": "#55A868",
        "no_early_exit": "#C44E52", "no_uncertainty": "#8172B2",
        "fixed_roles": "#937860", "oracle": "#CCB974",
    }
    labels = {
        "full": "AeroMesh (full)", "no_self_healing": "No self-healing",
        "no_priority": "No priority classes", "no_early_exit": "No early exit",
        "no_uncertainty": "No uncertainty policy", "fixed_roles": "Fixed roles (no auction)",
        "oracle": "Oracle central controller",
    }
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for ablation, color in ablation_colors.items():
        pts = [(d["phi_mean"], d["psi_norm"]) for d in data if d["ablation"] == ablation]
        phi = [p[0] for p in pts]
        psi = [p[1] for p in pts]
        ax.scatter(phi, psi, s=14, alpha=0.5, color=color, label=labels[ablation])
        ax.scatter([np.mean(phi)], [np.mean(psi)], s=140, marker="X", color=color,
                   edgecolor="black", linewidth=0.6, zorder=5)
    ax.set_xlabel(r"Mean priority-weighted connectivity $\Phi$")
    ax.set_ylabel(r"Mean assessment value $\Psi$ (normalized to AeroMesh)")
    ax.set_title("Figure B. Joint connectivity-assessment frontier\n(co-simulator, reduced-fidelity role-coupling model)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cosim_pareto.pdf")
    fig.savefig(f"{OUT_DIR}/cosim_pareto.svg")
    print("wrote cosim_pareto.pdf/svg")


if __name__ == "__main__":
    figure_a()
    figure_b()
