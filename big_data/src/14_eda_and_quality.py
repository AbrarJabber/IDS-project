"""
UNSW-NB15: five EDA scenarios and data-quality checks.

Inputs:
    data/raw/UNSW_NB15_training-set.csv
    data/raw/UNSW_NB15_testing-set.csv
    data/full/UNSW-NB15_1.csv through UNSW-NB15_4.csv
    data/full/NUSW-NB15_features.csv

Outputs:
    big_data/results/eda/<timestamp>/

Raw data and previous results are never modified.
"""

import argparse
import csv
import json
import os
import platform
import sys

from datetime import datetime, timezone
from pathlib import Path


NUMERIC_FEATURES = [
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
]


def save_csv(output_dir, filename, rows, fieldnames=None):
    """Save small aggregated results using Python CSV writing."""

    converted = [
        row.asDict() if hasattr(row, "asDict") else row
        for row in rows
    ]

    if fieldnames is None:
        if not converted:
            raise ValueError(
                f"No results or column names provided for {filename}"
            )

        fieldnames = list(converted[0].keys())

    output_path = output_dir / filename

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(converted)

    print(f"Saved: {filename}", flush=True)


def normalize_for_eda(df):
    """Normalize selected fields for descriptive analysis."""

    from pyspark.sql import functions as F

    df = df.withColumn(
        "label",
        F.col("label").cast("int")
    )

    category = F.lower(
        F.trim(F.col("attack_cat"))
    )

    df = df.withColumn(
        "attack_cat",
        F.when(
            F.col("label") == 0,
            "normal"
        ).when(
            category == "backdoors",
            "backdoor"
        ).when(
            category.isNull() | (category == ""),
            "unknown_attack"
        ).otherwise(category)
    )

    for column_name in ["proto", "service", "state"]:

        if column_name not in df.columns:
            continue

        value = F.trim(F.col(column_name))

        df = df.withColumn(
            column_name,
            F.when(
                value.isNull()
                | (value == "")
                | (value == "-"),
                "unknown"
            ).otherwise(value)
        )

    for column_name in NUMERIC_FEATURES:

        if column_name in df.columns:
            df = df.withColumn(
                column_name,
                F.col(column_name).cast("double")
            )

    return df


def analyze_missing_values(raw_df, dataset_name, output_dir):
    """Count null/empty fields and dash placeholders separately."""

    from pyspark.sql import functions as F

    expressions = []

    for index, field in enumerate(raw_df.schema.fields):

        value = F.col(field.name)

        text_value = F.trim(
            value.cast("string")
        )

        expressions.extend([
            F.sum(
                F.when(
                    value.isNull() | (text_value == ""),
                    1
                ).otherwise(0)
            ).alias(f"missing_{index}"),

            F.sum(
                F.when(
                    text_value == "-",
                    1
                ).otherwise(0)
            ).alias(f"dash_{index}")
        ])

    result = raw_df.agg(
        F.count("*").alias("records"),
        *expressions
    ).first().asDict()

    total_records = result["records"]

    if total_records == 0:
        raise ValueError(
            f"The {dataset_name} dataset is empty."
        )

    rows = []

    for index, field in enumerate(raw_df.schema.fields):

        missing_count = result[f"missing_{index}"]

        rows.append({
            "dataset": dataset_name,
            "column": field.name,
            "spark_type": field.dataType.simpleString(),
            "records": total_records,
            "null_or_empty": missing_count,
            "missing_percent": (
                100 * missing_count / total_records
            ),
            "dash_placeholders": result[f"dash_{index}"]
        })

    save_csv(
        output_dir,
        f"quality_{dataset_name}.csv",
        rows
    )

    return total_records


def analyze_numeric_quality(raw_df, dataset_name, output_dir):
    """Check selected numeric fields before computing statistics."""

    from pyspark.sql import functions as F

    rows = []

    for feature in NUMERIC_FEATURES:

        if feature not in raw_df.columns:
            continue

        original = F.col(feature)
        numeric = original.cast("double")

        present = (
            original.isNotNull()
            & (F.trim(original.cast("string")) != "")
        )

        result = raw_df.agg(
            F.sum(
                F.when(
                    present & numeric.isNull(),
                    1
                ).otherwise(0)
            ).alias("invalid_cast"),

            F.sum(
                F.when(
                    F.isnan(numeric),
                    1
                ).otherwise(0)
            ).alias("nan"),

            F.sum(
                F.when(
                    F.abs(numeric) == float("inf"),
                    1
                ).otherwise(0)
            ).alias("infinite"),

            F.sum(
                F.when(
                    numeric < 0,
                    1
                ).otherwise(0)
            ).alias("negative")
        ).first().asDict()

        rows.append({
            "dataset": dataset_name,
            "feature": feature,
            **result
        })

    save_csv(
        output_dir,
        f"numeric_quality_{dataset_name}.csv",
        rows,
        [
            "dataset",
            "feature",
            "invalid_cast",
            "nan",
            "infinite",
            "negative"
        ]
    )


def analyze_dataset(raw_df, dataset_name, output_dir):
    """Create the aggregate tables supporting the five EDA scenarios."""

    from pyspark.sql import functions as F

    print(
        f"\n=== ANALYZING {dataset_name.upper()} ===",
        flush=True
    )

    total_records = analyze_missing_values(
        raw_df,
        dataset_name,
        output_dir
    )

    analyze_numeric_quality(
        raw_df,
        dataset_name,
        output_dir
    )

    required_columns = [
        "label",
        "attack_cat",
        "proto",
        "service"
    ]

    missing_columns = [
        name
        for name in required_columns
        if name not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing required columns "
            f"{missing_columns}"
        )

    selected_columns = required_columns + [
        name
        for name in NUMERIC_FEATURES
        if name in raw_df.columns
    ]

    df = normalize_for_eda(
        raw_df.select(*selected_columns)
    ).cache()

    try:
        invalid_labels = df.filter(
            F.col("label").isNull()
            | ~F.col("label").isin(0, 1)
        ).count()

        if invalid_labels:
            raise ValueError(
                f"{dataset_name}: found "
                f"{invalid_labels} invalid labels."
            )

        # SCENARIO 1: Normal versus attack proportions
        class_counts = (
            df.groupBy("label")
            .count()
            .orderBy("label")
            .collect()
        )

        class_rows = [
            {
                "dataset": dataset_name,
                "label": row["label"],
                "records": row["count"],
                "percent": (
                    100 * row["count"] / total_records
                )
            }
            for row in class_counts
        ]

        save_csv(
            output_dir,
            f"class_distribution_{dataset_name}.csv",
            class_rows
        )

        # SCENARIO 2: Attack-family distribution
        category_counts = (
            df.groupBy("attack_cat")
            .count()
            .orderBy(F.desc("count"))
            .collect()
        )

        category_rows = [
            {
                "dataset": dataset_name,
                "attack_cat": row["attack_cat"],
                "records": row["count"],
                "percent": (
                    100 * row["count"] / total_records
                )
            }
            for row in category_counts
        ]

        save_csv(
            output_dir,
            f"attack_categories_{dataset_name}.csv",
            category_rows
        )

        # SCENARIO 3: Protocol/service attack rates
        for feature in ["proto", "service"]:

            grouped = (
                df.groupBy(feature)
                .agg(
                    F.count("*").alias("records"),
                    F.sum("label").alias("attacks")
                )
                .withColumn(
                    "attack_rate",
                    F.col("attacks") / F.col("records")
                )
                .orderBy(F.desc("records"))
            )

            save_csv(
                output_dir,
                f"{feature}_attack_rates_{dataset_name}.csv",
                grouped.collect()
            )

        # SCENARIO 4: Numerical characteristics by class
        numeric_rows = []

        for feature in NUMERIC_FEATURES:

            if feature not in df.columns:
                continue

            # Exclude invalid values only from these summaries.
            # No records are removed from the source datasets.
            valid = df.filter(
                F.col(feature).isNotNull()
                & ~F.isnan(F.col(feature))
                & (F.abs(F.col(feature)) != float("inf"))
                & (F.col(feature) >= 0)
            )

            statistics = (
                valid.groupBy("label")
                .agg(
                    F.count("*").alias("valid_records"),
                    F.min(feature).alias("minimum"),
                    F.max(feature).alias("maximum"),
                    F.avg(feature).alias("mean"),
                    F.stddev_samp(feature).alias("stddev"),
                    F.percentile_approx(
                        feature,
                        [0.25, 0.50, 0.75, 0.95, 0.99],
                        10000
                    ).alias("quantiles")
                )
                .orderBy("label")
                .collect()
            )

            for row in statistics:

                values = row.asDict()
                quantiles = values.pop("quantiles")

                numeric_rows.append({
                    "dataset": dataset_name,
                    "feature": feature,
                    **values,
                    "p25": quantiles[0],
                    "p50": quantiles[1],
                    "p75": quantiles[2],
                    "p95": quantiles[3],
                    "p99": quantiles[4]
                })

        save_csv(
            output_dir,
            f"numeric_summary_{dataset_name}.csv",
            numeric_rows,
            [
                "dataset",
                "feature",
                "label",
                "valid_records",
                "minimum",
                "maximum",
                "mean",
                "stddev",
                "p25",
                "p50",
                "p75",
                "p95",
                "p99"
            ]
        )

    finally:
        df.unpersist()

    return {
        "dataset": dataset_name,
        "records": total_records,
        "columns": len(raw_df.columns)
    }


def audit_duplicates(train_df, test_df, output_dir):
    """SCENARIO 5: duplicates, conflicting labels, and split overlap."""

    from pyspark.sql import functions as F

    print(
        "\n=== DUPLICATE AND TRAIN/TEST OVERLAP AUDIT ===",
        flush=True
    )

    if set(train_df.columns) != set(test_df.columns):
        raise ValueError(
            "Training and testing column names differ."
        )

    predictor_columns = sorted(
        name
        for name in train_df.columns
        if name not in ["id", "label", "attack_cat"]
    )

    training_types = {
        field.name: field.dataType
        for field in train_df.schema.fields
    }

    cached_keys = []
    within_split_rows = []

    try:
        for dataset_name, raw_df in [
            ("train", train_df),
            ("test", test_df)
        ]:

            typed = raw_df.select(*[
                F.col(name)
                .cast(training_types[name])
                .alias(name)
                for name in train_df.columns
            ])

            # Match the cleaning used by the existing models.
            typed = typed.withColumn(
                "service",
                F.when(
                    F.col("service") == "-",
                    "unknown"
                ).otherwise(F.col("service"))
            )

            # Use all predictors, in a fixed order, including nulls.
            predictor_json = F.to_json(
                F.struct(*[
                    F.col(name)
                    for name in predictor_columns
                ]),
                {"ignoreNullFields": "false"}
            )

            keys = typed.select(
                F.sha2(
                    predictor_json,
                    256
                ).alias("predictor_hash"),
                "label"
            ).cache()

            cached_keys.append(keys)

            total_records = keys.count()

            groups = (
                keys.groupBy("predictor_hash")
                .agg(
                    F.count("*").alias("records"),
                    F.countDistinct("label").alias("label_count")
                )
                .cache()
            )

            try:
                unique_predictors = groups.count()

                conflicting_groups = groups.filter(
                    F.col("label_count") > 1
                ).count()

                # This reproduces script 02's definition:
                # duplicate records considering every field except id.
                unique_records_without_id = (
                    raw_df.drop("id")
                    .distinct()
                    .count()
                )

                within_split_rows.append({
                    "dataset": dataset_name,
                    "records": total_records,
                    "duplicate_records_excluding_id": (
                        total_records - unique_records_without_id
                    ),
                    "unique_predictor_vectors": unique_predictors,
                    "extra_predictor_duplicates": (
                        total_records - unique_predictors
                    ),
                    "conflicting_predictor_groups": conflicting_groups
                })

            finally:
                groups.unpersist()

        train_keys = cached_keys[0].withColumn(
            "split",
            F.lit("train")
        )

        test_keys = cached_keys[1].withColumn(
            "split",
            F.lit("test")
        )

        combined = (
            train_keys.unionByName(test_keys)
            .groupBy("predictor_hash")
            .agg(
                F.sum(
                    F.when(
                        F.col("split") == "train",
                        1
                    ).otherwise(0)
                ).alias("train_rows"),

                F.sum(
                    F.when(
                        F.col("split") == "test",
                        1
                    ).otherwise(0)
                ).alias("test_rows"),

                F.countDistinct(
                    "label"
                ).alias("label_count")
            )
        )

        shared = combined.filter(
            (F.col("train_rows") > 0)
            & (F.col("test_rows") > 0)
        )

        overlap = shared.agg(
            F.count("*").alias(
                "shared_predictor_vectors"
            ),

            F.sum("train_rows").alias(
                "train_rows_with_shared_predictors"
            ),

            F.sum("test_rows").alias(
                "test_rows_with_shared_predictors"
            ),

            F.sum(
                F.when(
                    F.col("label_count") > 1,
                    1
                ).otherwise(0)
            ).alias(
                "shared_vectors_with_conflicting_labels"
            )
        ).first().asDict()

        overlap = {
            name: value if value is not None else 0
            for name, value in overlap.items()
        }

        save_csv(
            output_dir,
            "duplicate_audit.csv",
            within_split_rows
        )

        save_csv(
            output_dir,
            "train_test_overlap.csv",
            [overlap]
        )

        print("\nWithin-split results:")

        for row in within_split_rows:
            print(row)

        print("\nCross-split overlap:")
        print(overlap)

        return {
            "within_split": within_split_rows,
            "cross_split": overlap,
            "predictor_columns": predictor_columns,
            "method": (
                "SHA-256 of ordered typed predictor JSON; "
                "null fields included; baseline service cleaning."
            ),
            "interpretation": (
                "Shared predictor values do not prove identical "
                "captured network events. No records were removed."
            )
        }

    finally:
        for keys in cached_keys:
            keys.unpersist()


def main():

    parser = argparse.ArgumentParser(
        description="UNSW-NB15 EDA and data-quality analysis"
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Analyze only the official training/testing files."
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    project_dir = args.project_dir.resolve()

    raw_dir = project_dir / "data" / "raw"
    full_dir = project_dir / "data" / "full"

    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    full_paths = [
        full_dir / f"UNSW-NB15_{number}.csv"
        for number in range(1, 5)
    ]

    features_path = full_dir / "NUSW-NB15_features.csv"

    required_paths = [train_path, test_path]

    if not args.skip_full:
        required_paths += full_paths + [features_path]

    missing_files = [
        str(path)
        for path in required_paths
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing dataset files:\n"
            + "\n".join(missing_files)
        )

    os.environ.setdefault(
        "PYSPARK_PYTHON",
        sys.executable
    )

    import pyspark
    from pyspark.sql import SparkSession

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        project_dir
        / "big_data"
        / "results"
        / "eda"
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    manifest_path = output_dir / "run_manifest.json"

    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "spark": pyspark.__version__,
        "platform": platform.platform(),
        "workers": args.workers,
        "full_dataset_included": not args.skip_full,
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size
            }
            for path in required_paths
        ],
        "notes": [
            "EDA is descriptive; do not tune models on test findings.",
            "Approximate quantile accuracy is 10000.",
            "No raw records are removed or modified.",
            "Full-data packet columns differ from reduced splits.",
            "Null attack_cat for normal records can be structural."
        ]
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    spark = None

    try:
        spark = (
            SparkSession.builder
            .appName("UNSW-NB15-EDA-and-Quality")
            .master(f"local[{args.workers}]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.ansi.enabled", "false")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        manifest["spark_settings"] = {
            name: spark.conf.get(name)
            for name in [
                "spark.sql.shuffle.partitions",
                "spark.sql.adaptive.enabled",
                "spark.sql.ansi.enabled"
            ]
        }

        print("\nLoading official training/testing files...")
        print("Output folder:", output_dir, flush=True)

        train_df = spark.read.csv(
            str(train_path),
            header=True,
            inferSchema=True,
            mode="FAILFAST"
        )

        # Use the same schema for both official splits.
        test_df = spark.read.csv(
            str(test_path),
            header=True,
            schema=train_df.schema,
            mode="FAILFAST"
        )

        dataset_sizes = []

        dataset_sizes.append(
            analyze_dataset(
                train_df,
                "train",
                output_dir
            )
        )

        dataset_sizes.append(
            analyze_dataset(
                test_df,
                "test",
                output_dir
            )
        )

        manifest["duplicate_audit"] = audit_duplicates(
            train_df,
            test_df,
            output_dir
        )

        if not args.skip_full:

            print(
                "\nLoading the four original dataset files...",
                flush=True
            )

            with features_path.open(
                "r",
                encoding="cp1252",
                newline=""
            ) as file:

                reader = csv.DictReader(file)

                if "Name" not in (reader.fieldnames or []):
                    raise ValueError(
                        "Feature description must contain "
                        "a 'Name' column."
                    )

                feature_names = [
                    row["Name"].strip().replace(" ", "")
                    for row in reader
                ]

            feature_names = [
                "label" if name.lower() == "label" else name
                for name in feature_names
            ]

            if (
                len(feature_names) != 49
                or len(set(feature_names)) != 49
            ):
                raise ValueError(
                    "Expected 49 unique full-data feature names."
                )

            full_df = spark.read.csv(
                [str(path) for path in full_paths],
                header=False,
                inferSchema=False,
                mode="FAILFAST"
            )

            if len(full_df.columns) != 49:
                raise ValueError(
                    "Expected 49 columns in the original CSV files."
                )

            full_df = full_df.toDF(*feature_names)

            dataset_sizes.append(
                analyze_dataset(
                    full_df,
                    "full",
                    output_dir
                )
            )

        save_csv(
            output_dir,
            "dataset_sizes.csv",
            dataset_sizes
        )

        manifest.update({
            "status": "complete",
            "datasets": dataset_sizes,
            "finished_utc": (
                datetime.now(timezone.utc).isoformat()
            )
        })

        print("\n====================================")
        print("EDA AND QUALITY ANALYSIS COMPLETED")
        print("====================================")
        print("Results saved to:", output_dir)

    except Exception as error:

        manifest.update({
            "status": "failed",
            "error": str(error)
        })

        raise

    finally:

        manifest_path.write_text(
            json.dumps(manifest, indent=4),
            encoding="utf-8"
        )

        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()