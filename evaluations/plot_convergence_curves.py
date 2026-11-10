#!/usr/bin/env python3
"""
Generate publication-quality convergence figures for the Federated SAM paper:
1. Federated Validation Convergence Curve (Centers 1-5 in-domain validation DSC)
   with an inset zoom on Rounds 10-20 and markers for Val-best vs. Final (Last) round.
2. Federated Training Loss Convergence (Two-panel figure):
   (a) Normalized sample-weighted global training loss (L_r / L_1) across methods.
   (b) Per-client training loss trajectories (C1-C5) for SAM-LossScale* + Scale Dual-Decoder.

Reads directly from:
- checkpoints/runs/<run_name>/metrics_history.csv
- logs/<log_file>.log

Saves vector PDF (for LaTeX) and 300-DPI PNG under reports/figures/.
"""
import csv
import os
import re
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Configure publication-grade Matplotlib styling
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # TrueType fonts in PDF for clean LaTeX embedding
    "ps.fonttype": 42,
})

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")

# Key 20-round methods (identical Cosine LR schedule, Fold 1, C1-C5 training)
METHODS = [
    {
        "label": "SAM-FedAvg (Dice)",
        "run_dir": "fsm_fedavg_20round",
        "log_file": "fed_fgs_fsm_fedavg_20round.log",
        "color": "#5C6B73",
        "linestyle": "--",
        "marker": "o",
        "lw": 1.8,
        "zorder": 2,
    },
    {
        "label": "SAM-FedAvg + Comp. Loss",
        "run_dir": "ablation_sam_fedavg_composite_loss_r8_20rounds",
        "log_file": "fed_fgs_ablation_sam_fedavg_composite_loss_r8_20rounds.log",
        "color": "#8D99AE",
        "linestyle": ":",
        "marker": "v",
        "lw": 1.8,
        "zorder": 2,
    },
    {
        "label": "SAM-LossScale* + Comp. Loss",
        "run_dir": "ablation_sam_fedgs_composite_loss_r8_20rounds",
        "log_file": "fed_fgs_ablation_sam_fedgs_composite_loss_r8_20rounds.log",
        "color": "#2B6CB0",
        "linestyle": "-",
        "marker": "s",
        "lw": 2.0,
        "zorder": 3,
    },
    {
        "label": "SAM-LossScale* + MoDE-4 (Token)",
        "run_dir": "mode4_sam_fedgs_r8",
        "log_file": "mode4_sam_fedgs.log",
        "color": "#2F855A",
        "linestyle": "-.",
        "marker": "^",
        "lw": 2.0,
        "zorder": 4,
    },
    {
        "label": "SAM-LossScale* + MoDE-4-CP",
        "run_dir": "sam_fedgs_composite_mode4_cp_no_dftht_20rounds",
        "log_file": "sam_fedgs_composite_mode4_cp_no_dftht_20rounds.log",
        "color": "#805AD5",
        "linestyle": "-",
        "marker": "D",
        "lw": 2.0,
        "zorder": 4,
    },
    {
        "label": "SAM-LossScale* + Scale Dual-Decoder",
        "run_dir": "sam_fedgs_scale_decoder_moe_r8_20rounds",
        "log_file": "sam_fedgs_scale_decoder_moe_r8_20rounds.log",
        "color": "#D62828",
        "linestyle": "-",
        "marker": "*",
        "lw": 2.5,
        "zorder": 5,
    },
    {
        "label": "Hierarchical Bi-Level MoE",
        "run_dir": "hierarchical_bi_level_moe_no_dftht_r8_20rounds",
        "log_file": "hierarchical_bi_level_moe_no_dftht_r8_20rounds.log",
        "color": "#D97706",
        "linestyle": "--",
        "marker": "P",
        "lw": 1.9,
        "zorder": 3,
    },
]


def load_val_dsc(run_dir: str) -> Tuple[List[int], List[float]]:
    """Read per-round in-domain validation DSC (Centers 1-5) from metrics_history.csv."""
    csv_path = os.path.join(REPO_ROOT, "checkpoints", "runs", run_dir, "metrics_history.csv")
    rounds, dscs = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            dscs.append(float(row["in_domain_dsc"]))
    return rounds, dscs


def load_train_losses(log_file: str) -> Tuple[List[int], List[float], Dict[str, List[float]]]:
    """
    Parse per-client training losses from logs/<log_file>, deduplicating resumed rounds.
    Returns:
      rounds: [1..20]
      weighted_losses: sample-weighted average training loss across C1-C5 per round
      per_client: {"C1": [...], ..., "C5": [...]}
    """
    log_path = os.path.join(REPO_ROOT, "logs", log_file)
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r"\[Communication Round\s+(\d+)/\d+\]", text)
    round_data: Dict[int, Tuple[float, Dict[str, float]]] = {}

    for i in range(1, len(blocks), 2):
        rd = int(blocks[i])
        body = blocks[i + 1]
        clients = re.findall(
            r"\[(C\d)\] Completed \d+ local epochs \((\d+) samples\) \| Loss:\s*([0-9.]+)",
            body,
        )
        if len(clients) == 5:
            tot_n = sum(int(n) for _, n, _ in clients)
            w_loss = sum(int(n) * float(loss) for _, n, loss in clients) / tot_n
            c_dict = {c: float(loss) for c, _, loss in clients}
            round_data[rd] = (w_loss, c_dict)

    sorted_rds = sorted(round_data.keys())
    weighted_losses = [round_data[r][0] for r in sorted_rds]
    per_client = {f"C{k}": [round_data[r][1][f"C{k}"] for r in sorted_rds] for k in range(1, 6)}
    return sorted_rds, weighted_losses, per_client


def plot_validation_convergence() -> None:
    """Figure 1: Federated Validation Convergence Curve (Centers 1-5) with Inset Zoom."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    # Inset zoom on Rounds 10-20 placed in the clean bottom-right region
    ax_ins = inset_axes(
        ax,
        width="53%",
        height="45%",
        loc="lower right",
        bbox_to_anchor=(0.0, 0.05, 0.98, 0.95),
        bbox_transform=ax.transAxes,
    )

    for m in METHODS:
        rounds, dscs = load_val_dsc(m["run_dir"])
        ms = 6.5 if m["marker"] == "*" else 4.2

        # Main plot line
        ax.plot(
            rounds,
            dscs,
            label=m["label"],
            color=m["color"],
            linestyle=m["linestyle"],
            marker=m["marker"],
            markersize=ms,
            linewidth=m["lw"],
            alpha=0.92,
            zorder=m["zorder"],
        )

        # Highlight Val-best round with an open halo ring
        best_idx = max(range(len(dscs)), key=lambda i: dscs[i])
        ax.scatter(
            [rounds[best_idx]],
            [dscs[best_idx]],
            s=80,
            facecolors="none",
            edgecolors=m["color"],
            linewidths=1.4,
            zorder=m["zorder"] + 2,
        )

        # Inset zoom plot line
        ax_ins.plot(
            rounds,
            dscs,
            color=m["color"],
            linestyle=m["linestyle"],
            marker=m["marker"],
            markersize=ms - 0.4,
            linewidth=m["lw"] * 0.9,
            alpha=0.92,
            zorder=m["zorder"],
        )
        ax_ins.scatter(
            [rounds[best_idx]],
            [dscs[best_idx]],
            s=70,
            facecolors="none",
            edgecolors=m["color"],
            linewidths=1.3,
            zorder=m["zorder"] + 2,
        )

    # Configure main axes
    ax.set_xlabel("Federated Communication Rounds")
    ax.set_ylabel("In-Domain Validation DSC (Centers 1–5 Mean)")
    ax.set_title("Federated Convergence on Seen-Center Validation Data (Fold 1)")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.48, 0.772)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)

    # Configure inset axes (Rounds 10-20)
    ax_ins.set_xlim(9.7, 20.3)
    ax_ins.set_ylim(0.670, 0.758)
    ax_ins.set_xticks([10, 12, 14, 16, 18, 20])
    ax_ins.set_yticks([0.68, 0.70, 0.72, 0.74])
    ax_ins.grid(True, linestyle="--", alpha=0.25, linewidth=0.6)
    ax_ins.tick_params(labelsize=8.0)
    ax_ins.set_title("Rounds 10–20 Zoom (Hollow rings = Val-best)", fontsize=8.3, pad=3, fontweight="normal")
    for spine in ax_ins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)

    mark_inset(ax, ax_ins, loc1=1, loc2=2, fc="none", ec="#777777", linestyle=":", linewidth=0.9)

    # Legend cleanly placed above the plot in 2 columns so curves remain unobstructed
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        columnspacing=1.1,
        handlelength=2.2,
    )

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = os.path.join(FIG_DIR, f"fig_val_convergence.{ext}")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_training_loss_convergence() -> None:
    """
    Figure 2: Two-panel Federated Training Loss Convergence:
      (a) Relative normalized global training loss (L_r / L_1) across methods.
      (b) Per-client local training loss trajectories (C1-C5) for Scale Dual-Decoder.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # --- Panel (a): Normalized Global Training Loss across methods ---
    for m in METHODS:
        rds, w_losses, _ = load_train_losses(m["log_file"])
        l0 = w_losses[0]
        norm_losses = [l / l0 for l in w_losses]
        ms = 6.0 if m["marker"] == "*" else 4.0

        ax1.plot(
            rds,
            norm_losses,
            label=m["label"],
            color=m["color"],
            linestyle=m["linestyle"],
            marker=m["marker"],
            markersize=ms,
            linewidth=m["lw"],
            alpha=0.92,
            zorder=m["zorder"],
        )

    ax1.set_xlabel("Federated Communication Rounds")
    ax1.set_ylabel(r"Normalized Training Loss ($\mathcal{L}_r / \mathcal{L}_1$)")
    ax1.set_title("(a) Global Training Loss Convergence Across Methods")
    ax1.set_xlim(0.5, 20.5)
    ax1.set_ylim(0.15, 1.05)
    ax1.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#DDDDDD")

    # --- Panel (b): Per-Client Training Loss for Scale Dual-Decoder ---
    scale_m = [m for m in METHODS if "Scale Dual-Decoder" in m["label"]][0]
    rds, w_losses, per_client = load_train_losses(scale_m["log_file"])

    client_styles = {
        "C1": ("#2B6CB0", "-", "o", "Center 1 (n=201)"),
        "C2": ("#2F855A", "--", "s", "Center 2 (n=240)"),
        "C3": ("#805AD5", "-.", "^", "Center 3 (n=364)"),
        "C4": ("#D97706", ":", "D", "Center 4 (n=180)"),
        "C5": ("#D62828", "-", "v", "Center 5 (n=166)"),
    }
    for cid, (col, ls, mk, cname) in client_styles.items():
        ax2.plot(
            rds,
            per_client[cid],
            label=cname,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=4.0,
            linewidth=1.8,
            alpha=0.90,
        )

    ax2.plot(
        rds,
        w_losses,
        label="Weighted Global Mean",
        color="#1A202C",
        linestyle="-",
        linewidth=2.3,
        alpha=0.95,
        zorder=6,
    )

    ax2.set_xlabel("Federated Communication Rounds")
    ax2.set_ylabel(r"Local Multi-Task Loss ($\mathcal{L}_{\mathrm{scale}}$)")
    ax2.set_title("(b) Per-Client Loss Stability (Scale Dual-Decoder)")
    ax2.set_xlim(0.5, 20.5)
    ax2.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax2.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)
    ax2.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="white", edgecolor="#DDDDDD")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = os.path.join(FIG_DIR, f"fig_train_loss_convergence.{ext}")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_validation_convergence()
    plot_training_loss_convergence()
    print("Done generating all figures.")


if __name__ == "__main__":
    main()
