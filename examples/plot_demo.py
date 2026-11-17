"""Create the three-panel figure shown in the README."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def draw(report: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "axes.titleweight": "bold",
        "text.color": "#132f36",
        "axes.labelcolor": "#132f36",
        "xtick.color": "#4b6369",
        "ytick.color": "#4b6369",
        "axes.edgecolor": "#b6c7c8",
    })
    teal, blue, gray = "#087d71", "#496499", "#768589"
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9))
    fig.patch.set_facecolor("#f2f8f7")
    fig.subplots_adjust(left=.065, right=.978, bottom=.18, top=.73, wspace=.33)
    fig.text(.065, .91, "Clinical Model Validation Studio", fontsize=18, weight="bold")
    fig.text(.066, .84, "FICTIONAL DATA  /  STRATIFIED HOLDOUT  /  LOGISTIC REGRESSION", color="#426368", fontsize=8.5)

    ax = axes[0]
    roc = report["roc_curve"]
    ax.plot([0, 1], [0, 1], color=gray, linestyle=":", linewidth=1)
    ax.plot([r["fpr"] for r in roc], [r["tpr"] for r in roc], color=teal, linewidth=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title(f"ROC  |  AUC {report['metrics']['roc_auc']:.3f}", loc="left", pad=9)

    ax = axes[1]
    cal = report["calibration"]
    ax.plot([0, 1], [0, 1], color=gray, linestyle=":", linewidth=1)
    ax.plot([r["mean_predicted"] for r in cal], [r["observed_rate"] for r in cal],
            color=teal, marker="o", markersize=3, linewidth=1.4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted risk", ylabel="Observed fraction")
    ax.set_title("Calibration", loc="left", pad=9)

    ax = axes[2]
    dca = [r for r in report["decision_curve"] if r["threshold"] <= .45]
    x = [r["threshold"] for r in dca]
    for key, name, color, style in [
        ("net_benefit", "Model", teal, "-"),
        ("treat_all", "Treat all", blue, "--"),
        ("treat_none", "Treat none", gray, ":"),
    ]:
        ax.plot(x, [r[key] for r in dca], label=name, color=color, linestyle=style, linewidth=1.5)
    ax.set(xlabel="Risk threshold", ylabel="Net benefit")
    ax.set_title("Decision curve", loc="left", pad=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color="#e4eeee", linewidth=.7)
        ax.set_axisbelow(True)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(json.loads(args.report.read_text(encoding="utf-8")), args.output)
