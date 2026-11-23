"""
Create model figures and a calculated-results summary
from a completed script 16 run.

No Spark or model training is required.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True
    )

    args = parser.parse_args()
    source = args.input_dir.resolve()

    manifest = json.loads(
        (source / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("status") != "complete":
        raise ValueError(
            "Script 16 must finish successfully before plotting."
        )

    metrics = pd.read_csv(
        source / "model_metrics.csv"
    )

    categories = pd.read_csv(
        source / "rf_attack_category_results.csv"
    )

    timings = pd.read_csv(
        source / "model_training_times.csv"
    )

    threshold = float(
        manifest["selected_rf_threshold"]
    )

    if metrics.duplicated(
        ["model", "group", "threshold"]
    ).any():
        raise ValueError(
            "Duplicate evaluation rows found."
        )

    def get_row(model, group, cutoff):

        rows = metrics[
            (metrics["model"] == model)
            & (metrics["group"] == group)
            & np.isclose(
                metrics["threshold"],
                cutoff
            )
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected one row: "
                f"{model}, {group}, {cutoff}"
            )

        return rows.iloc[0]

    comparisons = [
        ("logistic_regression", 0.50, "LR 0.50"),
        ("random_forest", 0.50, "RF 0.50"),
        (
            "random_forest",
            threshold,
            f"RF {threshold:.2f}"
        )
    ]

    full = pd.DataFrame([
        get_row(model, "all_test", cutoff)
        for model, cutoff, _ in comparisons
    ])

    labels = [
        label
        for _, _, label in comparisons
    ]

    figures = source / "figures"
    figures.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    captions = []

    def save(name, caption):

        plt.tight_layout()

        plt.savefig(
            figures / name,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        captions.append(
            f"- **{name}**: {caption}"
        )

        print("Saved:", name)

    # -----------------------------------------------------
    # Figure 1: Full-test model comparison
    # -----------------------------------------------------

    fields = [
        "accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1"
    ]

    titles = [
        "Accuracy",
        "Attack precision",
        "Attack recall",
        "Attack F1"
    ]

    positions = np.arange(len(fields))

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for index, (_, row) in enumerate(
        full.iterrows()
    ):

        bars = ax.bar(
            positions + (index - 1) * 0.25,
            row[fields].to_numpy(dtype=float) * 100,
            width=0.25,
            label=labels[index]
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=2,
            fontsize=8
        )

    ax.set_xticks(positions, titles)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Metric (%)")

    ax.set_title(
        "Model comparison on the official test set"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3
    )

    save(
        "01_model_comparison.png",
        "Attack precision, recall and F1 are positive-class "
        "metrics. Both RF thresholds use the same fitted forest."
    )

    # -----------------------------------------------------
    # Figure 2: Confusion matrices
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4.5)
    )

    for ax, (_, row), title in zip(
        axes,
        full.iterrows(),
        labels
    ):

        counts = np.array([
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]]
        ], dtype=int)

        row_totals = counts.sum(
            axis=1,
            keepdims=True
        )

        percentages = np.divide(
            counts,
            row_totals,
            out=np.zeros((2, 2), dtype=float),
            where=row_totals != 0
        ) * 100

        ax.imshow(
            percentages,
            vmin=0,
            vmax=100,
            cmap="Blues"
        )

        for i in range(2):
            for j in range(2):

                text_color = (
                    "white"
                    if percentages[i, j] > 55
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{counts[i, j]:,}\n"
                    f"{percentages[i, j]:.2f}%",
                    ha="center",
                    va="center",
                    color=text_color
                )

        ax.set_xticks(
            [0, 1],
            ["Normal", "Attack"]
        )

        ax.set_yticks(
            [0, 1],
            ["Normal", "Attack"]
        )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(title)

    fig.suptitle(
        "Confusion matrices: counts and row percentages"
    )

    save(
        "02_confusion_matrices.png",
        "Rows represent actual classes. Colors share a "
        "0–100% scale. Percentages use each actual class total."
    )

    # -----------------------------------------------------
    # Figure 3: Overlap groups and their class composition
    # -----------------------------------------------------

    groups = [
        "all_test",
        "overlapping",
        "non_overlapping"
    ]

    group_labels = [
        "Entire test",
        "Overlapping",
        "Non-overlapping"
    ]

    cohort = pd.DataFrame([
        get_row("random_forest", group, threshold)
        for group in groups
    ])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    displayed_metrics = [
        ("accuracy", "Accuracy"),
        ("attack_f1", "Attack F1"),
        ("false_positive_rate", "False-positive rate")
    ]

    for index, (field, title) in enumerate(
        displayed_metrics
    ):

        bars = axes[0].bar(
            np.arange(3) + (index - 1) * 0.25,
            cohort[field] * 100,
            width=0.25,
            label=title
        )

        axes[0].bar_label(
            bars,
            fmt="%.1f",
            fontsize=8,
            padding=2
        )

    axes[0].set_xticks(
        range(3),
        group_labels
    )

    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Metric (%)")

    axes[0].set_title(
        f"RF at fixed threshold {threshold:.2f}"
    )

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=8
    )

    normal_percent = (
        100
        * cohort["normal_records"].to_numpy()
        / cohort["records"].to_numpy()
    )

    attack_percent = (
        cohort["attack_share"].to_numpy() * 100
    )

    axes[1].bar(
        group_labels,
        normal_percent,
        label="Normal",
        color="#31919c"
    )

    axes[1].bar(
        group_labels,
        attack_percent,
        bottom=normal_percent,
        label="Attack",
        color="#df7854"
    )

    for index in range(3):

        axes[1].text(
            index,
            normal_percent[index] / 2,
            f"{normal_percent[index]:.1f}%",
            ha="center"
        )

        axes[1].text(
            index,
            normal_percent[index]
            + attack_percent[index] / 2,
            f"{attack_percent[index]:.1f}%",
            ha="center"
        )

    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Share of records (%)")

    axes[1].set_title(
        "Class composition differs between groups"
    )

    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2
    )

    save(
        "03_overlap_comparison.png",
        "The full test set contains both groups. Differences "
        "are descriptive, not a causal overlap effect. "
        "Lower false-positive rate is better; higher accuracy "
        "and F1 are better."
    )

    # -----------------------------------------------------
    # Figure 4: Attack-family detection and misses
    # -----------------------------------------------------

    ordered = categories.sort_values([
        "attack_recall",
        "attack_cat"
    ])

    names = [
        f"{row['attack_cat']} "
        f"(n={int(row['total_attacks']):,})"
        for _, row in ordered.iterrows()
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6)
    )

    bars = axes[0].barh(
        names,
        ordered["attack_recall"] * 100,
        color="#31919c"
    )

    axes[0].bar_label(
        bars,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )

    axes[0].set_xlim(0, 112)
    axes[0].set_xlabel("Attack recall (%)")

    axes[0].set_title(
        "Detection by attack family"
    )

    bars = axes[1].barh(
        names,
        ordered["missed"],
        color="#df7854"
    )

    axes[1].bar_label(
        bars,
        padding=3
    )

    axes[1].set_xlim(
        0,
        max(1, int(ordered["missed"].max())) * 1.15
    )

    axes[1].set_xlabel(
        "Missed attacks (records)"
    )

    axes[1].set_title(
        "Missed attacks by family"
    )

    fig.suptitle(
        f"Random forest at threshold {threshold:.2f}: "
        "official test set"
    )

    save(
        "04_attack_family_results.png",
        "The model is binary. Family-level analysis uses "
        "known test annotations. Small families have "
        "limited sample support."
    )

    # -----------------------------------------------------
    # Summary loaded from actual saved result tables
    # -----------------------------------------------------

    def records(frame):

        return json.loads(
            frame.to_json(
                orient="records",
                double_precision=15
            )
        )

    summary = {
        "source_run": source.name,
        "run_manifest": manifest,
        "metrics": records(metrics),
        "pipeline_fit_times": records(timings),
        "attack_category_results": records(categories),
        "notes": [
            "This summary supersedes the manually entered "
            "model section in script 13's historical summary.",

            "Benchmark results remain in their original files.",

            "Training times are single observations, "
            "not repeated timing experiments.",

            "RF threshold was fixed from earlier validation; "
            "no threshold selection occurs here."
        ]
    }

    summary_path = (
        source / "verified_model_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=4,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    notes_path = (
        figures / "FIGURE_NOTES.md"
    )

    notes_path.write_text(
        "# Model figure notes\n\n"
        + "\n\n".join(captions),
        encoding="utf-8"
    )

    print("\n====================================")
    print("MODEL FIGURES COMPLETED")
    print("====================================")
    print("Number of figures:", len(captions))
    print("Figures:", figures)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()