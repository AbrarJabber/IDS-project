"""
Validate saved storage-benchmark results and generate report figures.
Requires pandas and matplotlib, already used by earlier visualizations.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


FORMATS = ["csv", "parquet", "delta"]
LABELS = ["CSV", "Parquet", "Delta"]
COLORS = ["#6B7280", "#0072B2", "#009E73"]

QUERIES = {
    "protocol_label": "All traffic: protocol and label",
    "tcp_label": "TCP traffic: label",
}


def save_figure(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"{name}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)
    print(f"Saved: {name}.png and {name}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()

    manifest = json.loads(
        (input_dir / "benchmark_manifest.json")
        .read_text(encoding="utf-8")
    )

    if manifest.get("status") != "passed":
        raise ValueError("The benchmark must have passed first.")

    runs = pd.read_csv(input_dir / "benchmark_runs.csv")
    saved_summary = pd.read_csv(input_dir / "benchmark_summary.csv")

    expected_groups = {
        (query, fmt)
        for query in QUERIES
        for fmt in FORMATS
    }

    actual_groups = set(zip(runs["query"], runs["format"]))

    if len(runs) != 36 or actual_groups != expected_groups:
        raise ValueError("Expected 36 runs across six query-format groups.")

    verified = runs["verified"].astype(str).str.lower().eq("true")
    if not verified.all():
        raise ValueError("Some benchmark runs were not verified.")

    if not all(
        math.isfinite(value) and value > 0
        for value in runs["seconds"]
    ):
        raise ValueError("Execution times must be finite and positive.")

    # Recalculate statistics from individual measurements.
    summary_rows = []

    for query in QUERIES:
        query_runs = runs[runs["query"] == query]

        if query_runs["result_sha256"].nunique() != 1:
            raise ValueError(f"Query result hashes differ: {query}")

        csv_mean = query_runs.loc[
            query_runs["format"] == "csv", "seconds"
        ].mean()

        for fmt in FORMATS:
            group = query_runs[query_runs["format"] == fmt]

            if sorted(group["repetition"].tolist()) != list(range(1, 7)):
                raise ValueError(f"Missing or repeated rounds: {query}/{fmt}")

            times = group["seconds"]
            mean = times.mean()
            sd = times.std(ddof=1)

            row = {
                "query": query,
                "format": fmt,
                "repetitions": len(times),
                "mean_seconds": mean,
                "stddev_seconds": sd,
                "median_seconds": times.median(),
                "min_seconds": times.min(),
                "max_seconds": times.max(),
                "speedup_vs_csv": csv_mean / mean,
                "coefficient_of_variation_percent": 100 * sd / mean,
            }

            previous = saved_summary[
                (saved_summary["query"] == query)
                & (saved_summary["format"] == fmt)
            ]

            if len(previous) != 1:
                raise ValueError(f"Invalid summary entry: {query}/{fmt}")

            for metric in (
                "mean_seconds",
                "stddev_seconds",
                "median_seconds",
                "speedup_vs_csv",
            ):
                if not math.isclose(
                    row[metric],
                    float(previous.iloc[0][metric]),
                    rel_tol=1e-8,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"Summary mismatch: {query}/{fmt}/{metric}"
                    )

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Use the storage measurements associated with this benchmark.
    conversion_dir = Path(manifest["conversion_directory"])
    conversion = json.loads(
        (conversion_dir / "full_conversion_manifest.json")
        .read_text(encoding="utf-8")
    )

    if conversion.get("status") != "passed":
        raise ValueError("The associated conversion did not pass.")

    if conversion["records"] != manifest["input_records"]:
        raise ValueError("Conversion and benchmark record counts differ.")

    sizes = pd.read_csv(conversion_dir / "storage_sizes.csv")

    if len(sizes) != 3 or set(sizes["format"]) != set(FORMATS):
        raise ValueError("Expected one storage-size row per format.")

    sizes = sizes.set_index("format").loc[FORMATS].copy()

    if not (sizes["records"] == manifest["input_records"]).all():
        raise ValueError("Storage measurements have different row counts.")

    sizes["size_mib"] = sizes["total_bytes"] / (1024 ** 2)
    csv_bytes = float(sizes.loc["csv", "total_bytes"])
    sizes["reduction_vs_csv_percent"] = (
        100 * (1 - sizes["total_bytes"] / csv_bytes)
    )

    figures = input_dir / "figures"
    figures.mkdir(exist_ok=True)

    summary.to_csv(
        input_dir / "verified_benchmark_summary.csv",
        index=False,
    )
    sizes.reset_index().to_csv(
        input_dir / "verified_storage_sizes.csv",
        index=False,
    )

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    })

    # Figure 1: storage size.
    fig, ax = plt.subplots(figsize=(8, 5))
    values = sizes["size_mib"].tolist()

    ax.bar(LABELS, values, color=COLORS, width=0.6)

    for position, value in enumerate(values):
        ax.text(
            position,
            value + max(values) * 0.025,
            f"{value:.2f}",
            ha="center",
        )

    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Stored file size (MiB)")
    ax.set_title("Full UNSW-NB15 storage comparison")
    ax.grid(axis="y", alpha=0.25)

    fig.text(
        0.5, 0.01,
        "Table directories include metadata and auxiliary files; "
        "1 MiB = 1,048,576 bytes.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, figures, "01_storage_sizes")

    # Figure 2: mean, sample SD, and all individual measurements.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    upper_limit = runs["seconds"].max() * 1.30
    offsets = [-0.10, -0.06, -0.02, 0.02, 0.06, 0.10]

    for ax, (query, title) in zip(axes, QUERIES.items()):
        selected = (
            summary[summary["query"] == query]
            .set_index("format")
            .loc[FORMATS]
        )

        means = selected["mean_seconds"].tolist()
        deviations = selected["stddev_seconds"].tolist()

        ax.bar(
            range(3),
            means,
            yerr=deviations,
            capsize=5,
            color=COLORS,
            width=0.6,
            alpha=0.85,
        )

        for position, fmt in enumerate(FORMATS):
            points = (
                runs[
                    (runs["query"] == query)
                    & (runs["format"] == fmt)
                ]
                .sort_values("repetition")["seconds"]
                .tolist()
            )

            ax.scatter(
                [position + offset for offset in offsets],
                points,
                color="black",
                s=22,
                zorder=3,
            )

            label_height = max(
                max(points),
                means[position] + deviations[position],
            )

            ax.text(
                position,
                label_height + upper_limit * 0.025,
                f"{means[position]:.3f}s",
                ha="center",
                fontsize=10,
            )

        ax.set_xticks(range(3))
        ax.set_xticklabels(LABELS)
        ax.set_title(title)
        ax.set_ylim(0, upper_limit)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Query execution time (seconds)")

    fig.text(
        0.5, 0.01,
        "Bars: means; error bars: ±1 sample SD; dots: all six measured runs. "
        "Warm-up runs excluded.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, figures, "02_query_times")

    # Figure 3: variation over successive benchmark rounds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (query, title) in zip(axes, QUERIES.items()):
        for fmt, label, color in zip(FORMATS, LABELS, COLORS):
            selected = runs[
                (runs["query"] == query)
                & (runs["format"] == fmt)
            ].sort_values("repetition")

            ax.plot(
                selected["repetition"],
                selected["seconds"],
                marker="o",
                color=color,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlabel("Benchmark round")
        ax.set_xticks(range(1, 7))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Query execution time (seconds)")

    fig.text(
        0.5, 0.01,
        "Format order varies by round. Lines connect successive "
        "measurements within one Spark session.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, figures, "03_runtime_by_round")

    print("\nSTORAGE VISUALIZATIONS COMPLETED")
    print("Figures:", figures)
    print("Number of figures: 3, each saved as PNG and PDF")
    print("\nVerified benchmark summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
    