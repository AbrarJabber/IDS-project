
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_DIR / "big_data" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_chart(filename):

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_cached_aggregation():

    df = pd.read_csv(
        RESULTS_DIR / "partition_benchmark.csv"
    ).sort_values("partitions")

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["partitions"].astype(str),
        df["average_seconds"],
        yerr=df["std_seconds"],
        capsize=5
    )

    plt.xlabel("Number of partitions")
    plt.ylabel("Average execution time (seconds)")
    plt.title(
        "Cached Spark Aggregation Performance"
    )

    save_chart("01_cached_aggregation.png")


def plot_end_to_end():

    original = pd.read_csv(
        RESULTS_DIR
        / "end_to_end_benchmark_original.csv"
    )

    reversed_order = pd.read_csv(
        RESULTS_DIR
        / "end_to_end_benchmark.csv"
    )

    comparison = original.merge(
        reversed_order,
        on="partitions",
        suffixes=("_original", "_reversed")
    ).sort_values("partitions")

    x = range(len(comparison))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - width / 2 for i in x],
        comparison["average_seconds_original"],
        width=width,
        label="Original order"
    )

    plt.bar(
        [i + width / 2 for i in x],
        comparison["average_seconds_reversed"],
        width=width,
        label="Reversed order"
    )

    plt.xticks(
        list(x),
        comparison["partitions"].astype(str)
    )

    plt.xlabel("Number of partitions")
    plt.ylabel("Average execution time (seconds)")
    plt.title("Effect of Benchmark Execution Order")
    plt.legend()

    save_chart("02_execution_order.png")


def plot_scalability():

    df = pd.read_csv(
        RESULTS_DIR / "scalability_results.csv"
    )

    summary = (
        df.groupby(
            ["workers", "fraction"],
            as_index=False
        )["training_seconds"]
        .mean()
    )

    plt.figure(figsize=(9, 5))

    for workers in sorted(
        summary["workers"].unique()
    ):

        subset = summary[
            summary["workers"] == workers
        ].sort_values("fraction")

        plt.plot(
            subset["fraction"] * 100,
            subset["training_seconds"],
            marker="o",
            label=f"{workers} worker threads"
        )

    plt.xlabel("Dataset fraction (%)")
    plt.ylabel("Average training time (seconds)")
    plt.title("Random Forest Training Scalability")
    plt.legend()
    plt.grid(alpha=0.3)

    save_chart("03_training_scalability.png")


def main():

    plot_cached_aggregation()
    plot_end_to_end()
    plot_scalability()

    print("\nPERFORMANCE VISUALIZATIONS COMPLETED")
    print("Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()

