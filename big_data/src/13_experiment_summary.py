
from pathlib import Path
import json
import platform
import sys
import statistics

import pandas as pd
import numpy as np
import pyspark
import matplotlib


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "big_data" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_benchmarks():

    scalability = pd.read_csv(
        RESULTS_DIR / "scalability_results.csv"
    )

    partition = pd.read_csv(
        RESULTS_DIR / "partition_benchmark.csv"
    )

    end_to_end = pd.read_csv(
        RESULTS_DIR / "end_to_end_benchmark.csv"
    )

    return scalability, partition, end_to_end


def get_scalability_summary(df):

    grouped = (
        df.groupby(
            ["workers", "fraction"]
        )
        .agg(
            average_training_seconds=(
                "training_seconds", "mean"
            ),
            average_total_seconds=(
                "total_seconds", "mean"
            ),
            repetitions=(
                "repetition", "count"
            )
        )
        .reset_index()
    )

    return grouped.round(4).to_dict(
        orient="records"
    )


def get_partition_summary(df):

    return (
        df[
            [
                "partitions",
                "average_seconds",
                "std_seconds"
            ]
        ]
        .sort_values("partitions")
        .to_dict(orient="records")
    )


def get_end_to_end_summary(df):

    return (
        df[
            [
                "partitions",
                "average_seconds",
                "median_seconds",
                "std_seconds"
            ]
        ]
        .sort_values("partitions")
        .to_dict(orient="records")
    )


def main():

    scalability, partition, end_to_end = (
        load_benchmarks()
    )

    summary = {
        "project": (
            "Big Data Analytics for Network "
            "Intrusion Detection Using Apache Spark"
        ),

        "environment": {
            "operating_system": platform.platform(),
            "python": sys.version.split()[0],
            "pyspark": pyspark.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__
        },

        "datasets": {
            "official_training_records": 175341,
            "official_testing_records": 82332,
            "full_dataset_records": 2540047,
            "full_dataset_columns": 49
        },

        "models": {
            "logistic_regression": {
                "accuracy": 0.8000,
                "roc_auc": 0.9417,
                "false_positives": 15335,
                "false_negatives": 1134
            },

            "random_forest_threshold_050": {
                "accuracy": 0.8122,
                "attack_precision": 0.7458,
                "attack_recall": 0.9996,
                "attack_f1": 0.8543,
                "roc_auc": 0.9743,
                "false_positives": 15442,
                "false_negatives": 17
            },

            "random_forest_threshold_060": {
                "accuracy": 0.8457,
                "attack_precision": 0.7855,
                "attack_recall": 0.9901,
                "attack_f1": 0.8760,
                "roc_auc": 0.9743,
                "false_positives": 12257,
                "false_negatives": 449
            }
        },

        "scalability": get_scalability_summary(
            scalability
        ),

        "cached_aggregation": get_partition_summary(
            partition
        ),

        "end_to_end_processing": get_end_to_end_summary(
            end_to_end
        ),

        "limitations": [
            "Experiments use local Spark execution, "
            "not a distributed multi-node cluster.",

            "Dataset sampling differs slightly "
            "between worker configurations.",

            "Some measurements were affected "
            "by warm-up and execution order.",

            "The full dataset was used for "
            "processing benchmarks, not model "
            "evaluation.",

            "The selected decision threshold "
            "was tuned on validation data."
        ]
    }

    json_file = (
        RESULTS_DIR / "experiment_summary.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    text_file = (
        RESULTS_DIR / "experiment_summary.txt"
    )

    with open(
        text_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            summary["project"] + "\n"
        )

        file.write("=" * 65 + "\n\n")

        file.write("ENVIRONMENT\n")

        for key, value in summary[
            "environment"
        ].items():

            file.write(
                f"{key}: {value}\n"
            )

        file.write("\nDATASETS\n")

        for key, value in summary[
            "datasets"
        ].items():

            file.write(
                f"{key}: {value}\n"
            )

        file.write("\nMODEL RESULTS\n")

        for name, metrics in summary[
            "models"
        ].items():

            file.write(
                f"\n{name}\n"
            )

            for metric, value in metrics.items():
                file.write(
                    f"  {metric}: {value}\n"
                )

        file.write("\nSCALABILITY RESULTS\n")

        for result in summary["scalability"]:
            file.write(
                str(result) + "\n"
            )

        file.write(
            "\nCACHED AGGREGATION RESULTS\n"
        )

        for result in summary[
            "cached_aggregation"
        ]:
            file.write(
                str(result) + "\n"
            )

        file.write(
            "\nEND-TO-END RESULTS\n"
        )

        for result in summary[
            "end_to_end_processing"
        ]:
            file.write(
                str(result) + "\n"
            )

        file.write("\nLIMITATIONS\n")

        for limitation in summary[
            "limitations"
        ]:
            file.write(
                "- " + limitation + "\n"
            )

    print("\nEXPERIMENT SUMMARY COMPLETED")
    print("JSON:", json_file)
    print("TXT:", text_file)

    print("\nTotal scalability runs:", len(scalability))
    print(
        "Partition configurations:",
        len(partition)
    )
    print(
        "End-to-end configurations:",
        len(end_to_end)
    )


if __name__ == "__main__":
    main()
