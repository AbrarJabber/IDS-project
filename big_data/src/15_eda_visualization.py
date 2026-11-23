"""
Generate EDA figures from the aggregate CSV files created by script 14.

This script does not require Spark.
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
        description="Generate UNSW-NB15 EDA figures."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Completed timestamped output folder from script 14."
    )

    args = parser.parse_args()
    input_dir = args.input_dir.resolve()

    manifest_path = input_dir / "run_manifest.json"

    with manifest_path.open(
        encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    if manifest.get("status") != "complete":
        raise ValueError(
            "This EDA run is incomplete. "
            "Complete script 14 before generating figures."
        )

    datasets = [
        item["dataset"]
        for item in manifest["datasets"]
    ]

    figures_dir = input_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    figure_notes = []

    def save_figure(filename, description):

        plt.tight_layout()

        plt.savefig(
            figures_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        figure_notes.append(
            f"- **{filename}**: {description}"
        )

        print(f"Saved: {filename}")

    # ---------------------------------------------------------
    # SCENARIO 1: Class distribution
    # ---------------------------------------------------------

    class_data = pd.concat(
        [
            pd.read_csv(
                input_dir
                / f"class_distribution_{dataset}.csv"
            )
            for dataset in datasets
        ],
        ignore_index=True
    )

    class_table = (
        class_data
        .pivot(
            index="dataset",
            columns="label",
            values="percent"
        )
        .reindex(datasets)
        .fillna(0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={
            0: "Normal",
            1: "Attack"
        })
    )

    ax = class_table.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=["#31919c", "#df7854"],
        rot=0
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            label_type="center",
            fontsize=10
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Share of records (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Scenario 1: Normal and attack proportions"
    )

    save_figure(
        "01_class_distribution.png",
        "Each percentage uses the total number of records "
        "in its own dataset as the denominator."
    )

    # ---------------------------------------------------------
    # SCENARIO 2: Attack-family composition
    # ---------------------------------------------------------

    category_data = pd.concat(
        [
            pd.read_csv(
                input_dir
                / f"attack_categories_{dataset}.csv"
            ).assign(dataset=dataset)
            for dataset in datasets
        ],
        ignore_index=True
    )

    attacks = category_data[
        category_data["attack_cat"] != "normal"
    ].copy()

    if not attacks.empty:

        attacks["attack_percent"] = (
            100
            * attacks["records"]
            / attacks.groupby("dataset")[
                "records"
            ].transform("sum")
        )

        category_table = (
            attacks
            .pivot(
                index="attack_cat",
                columns="dataset",
                values="attack_percent"
            )
            .reindex(columns=datasets)
            .fillna(0)
        )

        category_table.plot(
            kind="barh",
            figsize=(10, 7)
        )

        plt.xlabel("Share within attack records (%)")
        plt.ylabel("Attack family")
        plt.title(
            "Scenario 2: Attack-family composition"
        )

        save_figure(
            "02_attack_families.png",
            "Normal traffic is excluded. Percentages use "
            "attack records as the denominator. Consult "
            "the CSV files for category counts."
        )

    # ---------------------------------------------------------
    # SCENARIOS 3 AND 4: Per-dataset figures
    # ---------------------------------------------------------

    for dataset in datasets:

        # SCENARIO 3: Protocol/service attack rates
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 6)
        )

        for ax, feature in zip(
            axes,
            ["proto", "service"]
        ):

            rates = pd.read_csv(
                input_dir
                / f"{feature}_attack_rates_{dataset}.csv",
                keep_default_na=False
            )

            # Select high-volume groups, not extreme rates.
            rates = (
                rates
                .sort_values(
                    "records",
                    ascending=False
                )
                .head(12)
                .sort_values("attack_rate")
            )

            labels = [
                f"{row[feature]} "
                f"(n={int(row['records']):,})"
                for _, row in rates.iterrows()
            ]

            ax.barh(
                labels,
                rates["attack_rate"] * 100,
                color="#526caa"
            )

            ax.set_xlim(0, 100)

            ax.set_xlabel(
                "Attack records within group (%)"
            )

            ax.set_title(
                f"Top 12 {feature} groups by volume"
            )

        fig.suptitle(
            f"Scenario 3: Protocol and service "
            f"associations — {dataset}"
        )

        save_figure(
            f"03_protocol_service_{dataset}.png",
            "Groups are selected by record count. "
            "Group sizes are shown beside their names. "
            "Associations do not establish causation."
        )

        # SCENARIO 4: Numerical distributions
        numeric = pd.read_csv(
            input_dir
            / f"numeric_summary_{dataset}.csv"
        )

        features = list(
            numeric["feature"].unique()
        )

        if features:

            fig, axes = plt.subplots(
                1,
                len(features),
                figsize=(3.2 * len(features), 5),
                squeeze=False
            )

            for ax, feature in zip(
                axes[0],
                features
            ):

                subset = numeric[
                    numeric["feature"] == feature
                ]

                for _, row in subset.iterrows():

                    label = int(row["label"])

                    color = (
                        "#31919c"
                        if label == 0
                        else "#df7854"
                    )

                    # Thick bar: 25th to 75th percentile.
                    ax.vlines(
                        label,
                        row["p25"],
                        row["p75"],
                        linewidth=8,
                        alpha=0.6,
                        color=color
                    )

                    # Black dot: median.
                    ax.scatter(
                        label,
                        row["p50"],
                        color="black",
                        zorder=3
                    )

                    # Blue triangle: 95th percentile.
                    ax.scatter(
                        label,
                        row["p95"],
                        marker="^",
                        color="#526caa",
                        zorder=3
                    )

                ax.set_xticks(
                    [0, 1],
                    ["Normal", "Attack"]
                )

                ax.set_xlim(-0.6, 1.6)

                # Allows zeros while displaying large values.
                ax.set_yscale(
                    "symlog",
                    linthresh=1
                )

                ax.set_ylim(bottom=0)
                ax.set_title(feature)

                ax.set_ylabel(
                    "Value (symlog scale)"
                )

            fig.suptitle(
                f"Scenario 4: Numeric characteristics — "
                f"{dataset}\n"
                "Bars: 25th–75th percentile | "
                "Dots: median | Triangles: 95th percentile"
            )

            save_figure(
                f"04_numeric_quantiles_{dataset}.png",
                "Approximate quantiles from Spark. "
                "The symlog scale includes zero. "
                "Features have different units; compare "
                "classes within each feature. Invalid, "
                "nonfinite and negative values were "
                "counted separately and excluded from "
                "these numerical summaries."
            )

    # ---------------------------------------------------------
    # SCENARIO 5: Missingness and placeholders
    # ---------------------------------------------------------

    quality = pd.concat(
        [
            pd.read_csv(
                input_dir
                / f"quality_{dataset}.csv"
            )
            for dataset in datasets
        ],
        ignore_index=True
    )

    quality["dash_percent"] = (
        100
        * quality["dash_placeholders"]
        / quality["records"]
    )

    affected_fields = sorted(
        quality.loc[
            (quality["missing_percent"] > 0)
            | (quality["dash_percent"] > 0),
            "column"
        ].unique()
    )

    if affected_fields:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                12,
                max(5, len(affected_fields) * 0.4)
            )
        )

        settings = [
            (
                "missing_percent",
                "Null or empty (%)"
            ),
            (
                "dash_percent",
                "Dash placeholder (%)"
            )
        ]

        for ax, (metric, title) in zip(
            axes,
            settings
        ):

            matrix = (
                quality
                .pivot(
                    index="column",
                    columns="dataset",
                    values=metric
                )
                .reindex(
                    index=affected_fields,
                    columns=datasets
                )
            )

            color_map = plt.get_cmap(
                "YlOrRd"
            ).copy()

            color_map.set_bad("#dddddd")

            image = ax.imshow(
                np.ma.masked_invalid(
                    matrix.to_numpy()
                ),
                vmin=0,
                vmax=100,
                aspect="auto",
                cmap=color_map
            )

            ax.set_xticks(
                range(len(datasets)),
                datasets
            )

            ax.set_yticks(
                range(len(affected_fields)),
                affected_fields
            )

            ax.set_title(title)

            fig.colorbar(
                image,
                ax=ax,
                shrink=0.7
            )

        fig.suptitle(
            "Scenario 5: Missingness and placeholders\n"
            "Gray indicates a field absent from that dataset"
        )

    else:

        fig, ax = plt.subplots(
            figsize=(8, 3)
        )

        ax.axis("off")

        ax.text(
            0.5,
            0.5,
            "No null/empty values or dash placeholders "
            "were found.",
            ha="center",
            va="center",
            wrap=True
        )

    save_figure(
        "05_data_quality.png",
        "Null attack_cat values for normal full-data "
        "records can be structural. Dash placeholders "
        "are counted separately and are not automatically "
        "treated as errors. Check numeric_quality CSVs "
        "for numeric parsing and validity issues."
    )

    # SCENARIO 5: Predictor duplicates
    duplicates = pd.read_csv(
        input_dir / "duplicate_audit.csv"
    )

    duplicates["duplicate_percent"] = (
        100
        * duplicates["extra_predictor_duplicates"]
        / duplicates["records"]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        duplicates["dataset"],
        duplicates["duplicate_percent"],
        color="#526caa"
    )

    ax.bar_label(
        bars,
        fmt="%.2f%%",
        padding=4
    )

    ax.set_ylim(0, 100)

    ax.set_ylabel(
        "Extra predictor duplicates / records (%)"
    )

    ax.set_title(
        "Scenario 5: Duplicate predictor rows"
    )

    save_figure(
        "05_predictor_duplicates.png",
        "Extra duplicates equal total records minus "
        "unique predictor vectors. This is not the "
        "percentage of all rows in repeated groups. "
        "Train/test overlap is reported separately."
    )

    # Display the overlap audit without creating new metrics.
    overlap = pd.read_csv(
        input_dir / "train_test_overlap.csv"
    )

    print("\n=== TRAIN/TEST OVERLAP ===")

    for column in overlap.columns:
        print(
            f"{column}: "
            f"{int(overlap.iloc[0][column]):,}"
        )

    notes_path = figures_dir / "FIGURE_NOTES.md"

    notes_path.write_text(
        "# EDA figure notes\n\n"
        + "\n\n".join(figure_notes),
        encoding="utf-8"
    )

    print("\n====================================")
    print("EDA VISUALIZATIONS COMPLETED")
    print("====================================")
    print("Number of figures:", len(figure_notes))
    print("Figures saved to:", figures_dir)


if __name__ == "__main__":
    main()