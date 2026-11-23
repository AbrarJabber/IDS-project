"""
Model evaluation and train/test predictor-overlap analysis.

Models:
    Logistic Regression: maxIter=30, regParam=0.01
    Random Forest: 50 trees, maxDepth=10, seed=42

Random Forest thresholds:
    0.50: reference threshold
    0.60: previously selected using validation data

No threshold selection or hyperparameter tuning is performed here.
"""

import csv
import json
import os
import platform
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

SELECTED_RF_THRESHOLD = 0.60

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state"
]

os.environ.setdefault(
    "PYSPARK_PYTHON",
    sys.executable
)

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator
)


def write_csv(path, rows, fieldnames):
    """Write aggregate results without Hadoop filesystem output."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def prepare_data(df):
    """Apply the same service replacement as the baseline scripts."""

    return df.withColumn(
        "service",
        F.when(
            F.col("service") == "-",
            "unknown"
        ).otherwise(F.col("service"))
    )


def add_predictor_hash(df, predictor_columns):
    """Match the predictor-hash definition used in script 14."""

    ordered_predictors = sorted(predictor_columns)

    serialized = F.to_json(
        F.struct(*[
            F.col(name)
            for name in ordered_predictors
        ]),
        {"ignoreNullFields": "false"}
    )

    return df.withColumn(
        "predictor_hash",
        F.sha2(serialized, 256)
    )


def create_pipeline(model_name, predictor_columns):

    numeric_columns = [
        name
        for name in predictor_columns
        if name not in CATEGORICAL_COLUMNS
    ]

    stages = []

    for name in CATEGORICAL_COLUMNS:

        stages.append(
            StringIndexer(
                inputCol=name,
                outputCol=name + "_index",
                handleInvalid="keep"
            )
        )

    stages.append(
        OneHotEncoder(
            inputCols=[
                name + "_index"
                for name in CATEGORICAL_COLUMNS
            ],
            outputCols=[
                name + "_encoded"
                for name in CATEGORICAL_COLUMNS
            ],
            handleInvalid="keep"
        )
    )

    assembled_column = (
        "unscaled_features"
        if model_name == "logistic_regression"
        else "features"
    )

    stages.append(
        VectorAssembler(
            inputCols=numeric_columns + [
                name + "_encoded"
                for name in CATEGORICAL_COLUMNS
            ],
            outputCol=assembled_column,
            handleInvalid="error"
        )
    )

    if model_name == "logistic_regression":

        stages.append(
            StandardScaler(
                inputCol="unscaled_features",
                outputCol="features",
                withStd=True,
                withMean=False
            )
        )

        stages.append(
            LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=30,
                regParam=0.01
            )
        )

    elif model_name == "random_forest":

        stages.append(
            RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=50,
                maxDepth=10,
                seed=42
            )
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return Pipeline(stages=stages)


def safe_divide(numerator, denominator):
    """Undefined metrics are saved as blank CSV / null JSON values."""

    if denominator == 0:
        return None

    return numerator / denominator


def calculate_metrics(
    predictions,
    model_name,
    group_name,
    threshold,
    roc_auc
):
    """
    Compute binary attack-class metrics.

    LR uses Spark's original prediction column.
    RF uses attack_probability >= threshold, matching script 07.
    """

    if model_name == "logistic_regression":

        evaluated = predictions.withColumn(
            "evaluated_prediction",
            F.col("default_prediction").cast("int")
        )

        prediction_rule = "Spark default prediction"

    else:

        evaluated = predictions.withColumn(
            "evaluated_prediction",
            F.when(
                F.col("attack_probability") >= threshold,
                1
            ).otherwise(0)
        )

        prediction_rule = (
            f"attack_probability >= {threshold:.2f}"
        )

    count_rows = (
        evaluated
        .groupBy("label", "evaluated_prediction")
        .count()
        .collect()
    )

    counts = {
        (
            int(row["label"]),
            int(row["evaluated_prediction"])
        ): int(row["count"])
        for row in count_rows
    }

    tn = counts.get((0, 0), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    tp = counts.get((1, 1), 0)

    total = tn + fp + fn + tp
    normal_records = tn + fp
    attack_records = tp + fn

    return {
        "model": model_name,
        "group": group_name,
        "threshold": threshold,
        "prediction_rule": prediction_rule,
        "records": total,
        "normal_records": normal_records,
        "attack_records": attack_records,
        "attack_share": safe_divide(
            attack_records,
            total
        ),
        "accuracy": safe_divide(
            tp + tn,
            total
        ),
        "attack_precision": safe_divide(
            tp,
            tp + fp
        ),
        "attack_recall": safe_divide(
            tp,
            tp + fn
        ),
        "attack_f1": safe_divide(
            2 * tp,
            2 * tp + fp + fn
        ),
        "false_positive_rate": safe_divide(
            fp,
            fp + tn
        ),
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def save_predictions(predictions, output_path):
    """
    Stream prediction rows to CSV.

    Only IDs, labels, categories, overlap flags and predictions
    are exported; raw predictor columns are not exported.
    """

    fields = [
        "id",
        "label",
        "attack_cat",
        "overlap",
        "default_prediction",
        "attack_probability"
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for row in predictions.select(
            *fields
        ).toLocalIterator():

            writer.writerow(row.asDict())


def save_rf_category_results(predictions, output_dir):

    evaluated = predictions.withColumn(
        "selected_prediction",
        F.when(
            F.col("attack_probability")
            >= SELECTED_RF_THRESHOLD,
            1
        ).otherwise(0)
    )

    category_results = (
        evaluated
        .filter(F.col("label") == 1)
        .groupBy("attack_cat")
        .agg(
            F.count("*").alias("total_attacks"),
            F.sum("selected_prediction").alias("detected")
        )
        .withColumn(
            "missed",
            F.col("total_attacks") - F.col("detected")
        )
        .withColumn(
            "attack_recall",
            F.col("detected") / F.col("total_attacks")
        )
        .orderBy("attack_recall", "attack_cat")
    )

    rows = [
        row.asDict()
        for row in category_results.collect()
    ]

    write_csv(
        output_dir / "rf_attack_category_results.csv",
        rows,
        [
            "attack_cat",
            "total_attacks",
            "detected",
            "missed",
            "attack_recall"
        ]
    )


def print_metric_summary(result):

    def display(value):
        return (
            "undefined"
            if value is None
            else f"{value:.4f}"
        )

    print(
        f"\n{result['model']} | "
        f"{result['group']} | "
        f"threshold={result['threshold']:.2f}"
    )

    print(
        f"Records: {result['records']:,} | "
        f"Normal: {result['normal_records']:,} | "
        f"Attack: {result['attack_records']:,}"
    )

    for field in [
        "accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "false_positive_rate",
        "roc_auc"
    ]:
        print(
            f"{field}: {display(result[field])}"
        )

    print(
        f"TN={result['tn']}, FP={result['fp']}, "
        f"FN={result['fn']}, TP={result['tp']}"
    )


def main():

    train_path = (
        DATA_DIR / "UNSW_NB15_training-set.csv"
    )

    test_path = (
        DATA_DIR / "UNSW_NB15_testing-set.csv"
    )

    for path in [train_path, test_path]:

        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        PROJECT_DIR
        / "big_data"
        / "results"
        / "model_audit"
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    manifest_path = output_dir / "run_manifest.json"

    manifest = {
        "status": "running",
        "started_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "python": sys.version.split()[0],
        "pyspark": pyspark.__version__,
        "platform": platform.platform(),
        "spark_master": "local[2]",
        "selected_rf_threshold": SELECTED_RF_THRESHOLD,
        "threshold_source": (
            "Previously selected by script 06 on validation "
            "data; fixed in this verification experiment."
        ),
        "model_parameters": {
            "logistic_regression": {
                "maxIter": 30,
                "regParam": 0.01
            },
            "random_forest": {
                "numTrees": 50,
                "maxDepth": 10,
                "seed": 42
            }
        },
        "notes": [
            "Models are refitted because earlier scripts "
            "did not persist fitted models.",
            "Each model is fitted once; the same fitted model "
            "is used for all test groups.",
            "No test-based threshold or hyperparameter selection.",
            "Overlapping predictors do not prove identical events.",
            "Group comparisons are descriptive, not a causal "
            "estimate of the effect of overlap.",
            "Undefined metrics are saved as null/blank.",
            "Training time measures pipeline fitting, including "
            "encoding and LR scaling."
        ]
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    spark = None
    test_with_overlap = None
    metric_rows = []
    timing_rows = []

    try:
        spark = (
            SparkSession.builder
            .appName("UNSW-NB15-Model-Overlap-Evaluation")
            .master("local[2]")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        print("\nLoading training and testing data...")
        print("Output folder:", output_dir, flush=True)

        train_raw = spark.read.csv(
            str(train_path),
            header=True,
            inferSchema=True,
            mode="FAILFAST"
        )

        test_raw = spark.read.csv(
            str(test_path),
            header=True,
            schema=train_raw.schema,
            mode="FAILFAST"
        )

        train_df = prepare_data(
            train_raw
        ).drop("id")

        test_df = prepare_data(test_raw)

        # Explicitly determine predictors before adding metadata.
        predictor_columns = [
            name
            for name in train_df.columns
            if name not in ["label", "attack_cat"]
        ]

        manifest["predictor_columns"] = predictor_columns

        train_count = train_df.count()
        test_count = test_df.count()

        for name, df in [
            ("training", train_df),
            ("testing", test_df)
        ]:

            invalid = df.filter(
                F.col("label").isNull()
                | ~F.col("label").isin(0, 1)
            ).count()

            if invalid:
                raise ValueError(
                    f"{name}: found {invalid} invalid labels."
                )

        print(f"Training records: {train_count:,}")
        print(f"Testing records: {test_count:,}")

        print(
            "\nIdentifying shared predictor patterns...",
            flush=True
        )

        train_hashes = (
            add_predictor_hash(
                train_df,
                predictor_columns
            )
            .select("predictor_hash")
            .distinct()
            .withColumn(
                "overlap",
                F.lit(True)
            )
        )

        test_with_overlap = (
            add_predictor_hash(
                test_df,
                predictor_columns
            )
            .join(
                train_hashes,
                on="predictor_hash",
                how="left"
            )
            .fillna({"overlap": False})
            .cache()
        )

        joined_count = test_with_overlap.count()

        if joined_count != test_count:
            raise ValueError(
                "Overlap join changed the number of test rows."
            )

        overlap_count = test_with_overlap.filter(
            F.col("overlap")
        ).count()

        non_overlap_count = (
            test_count - overlap_count
        )

        print(f"Overlapping test rows: {overlap_count:,}")
        print(
            f"Non-overlapping test rows: "
            f"{non_overlap_count:,}"
        )

        manifest["dataset_counts"] = {
            "training": train_count,
            "testing": test_count,
            "overlapping_test": overlap_count,
            "non_overlapping_test": non_overlap_count
        }

        # Validate membership against the completed script 14 audit.
        if (
            train_count != 175341
            or test_count != 82332
            or overlap_count != 8541
        ):
            raise ValueError(
                "Dataset/overlap counts differ from the completed "
                "EDA audit. Expected 175341 training rows, "
                "82332 test rows and 8541 overlapping test rows. "
                "Check inputs before comparing results."
            )

        for model_name in [
            "logistic_regression",
            "random_forest"
        ]:

            print(
                f"\n=== TRAINING {model_name.upper()} ===",
                flush=True
            )

            pipeline = create_pipeline(
                model_name,
                predictor_columns
            )

            training_start = time.perf_counter()

            fitted_model = pipeline.fit(train_df)

            training_seconds = (
                time.perf_counter() - training_start
            )

            predictions = (
                fitted_model.transform(test_with_overlap)
                .withColumn(
                    "attack_probability",
                    vector_to_array(
                        F.col("probability")
                    )[1]
                )
                .select(
                    "id",
                    "label",
                    "attack_cat",
                    "overlap",
                    F.col("prediction").alias(
                        "default_prediction"
                    ),
                    "attack_probability"
                )
                .cache()
            )

            try:
                if predictions.count() != test_count:
                    raise ValueError(
                        "Prediction count differs from test count."
                    )

                timing_rows.append({
                    "model": model_name,
                    "training_records": train_count,
                    "pipeline_fit_seconds": training_seconds
                })

                save_predictions(
                    predictions,
                    output_dir / f"{model_name}_predictions.csv"
                )

                thresholds = (
                    [0.50]
                    if model_name == "logistic_regression"
                    else [0.50, SELECTED_RF_THRESHOLD]
                )

                group_definitions = [
                    ("all_test", None),
                    ("overlapping", True),
                    ("non_overlapping", False)
                ]

                for group_name, overlap_flag in group_definitions:

                    if overlap_flag is None:
                        group_predictions = predictions
                    else:
                        group_predictions = predictions.filter(
                            F.col("overlap") == overlap_flag
                        )

                    class_counts = {
                        int(row["label"]): int(row["count"])
                        for row in (
                            group_predictions
                            .groupBy("label")
                            .count()
                            .collect()
                        )
                    }

                    # ROC-AUC requires both classes.
                    roc_auc = None

                    if (
                        class_counts.get(0, 0) > 0
                        and class_counts.get(1, 0) > 0
                    ):

                        evaluator = BinaryClassificationEvaluator(
                            labelCol="label",
                            rawPredictionCol="attack_probability",
                            metricName="areaUnderROC"
                        )

                        roc_auc = evaluator.evaluate(
                            group_predictions
                        )

                    for threshold in thresholds:

                        result = calculate_metrics(
                            group_predictions,
                            model_name,
                            group_name,
                            threshold,
                            roc_auc
                        )

                        metric_rows.append(result)
                        print_metric_summary(result)

                if model_name == "random_forest":
                    save_rf_category_results(
                        predictions,
                        output_dir
                    )

                # Save after each model so completed work is retained.
                write_csv(
                    output_dir / "model_metrics.csv",
                    metric_rows,
                    list(metric_rows[0].keys())
                )

                write_csv(
                    output_dir / "model_training_times.csv",
                    timing_rows,
                    list(timing_rows[0].keys())
                )

            finally:
                predictions.unpersist()

        # Check that group confusion matrices add to the full test.
        for model_name, threshold in [
            ("logistic_regression", 0.50),
            ("random_forest", 0.50),
            ("random_forest", SELECTED_RF_THRESHOLD)
        ]:

            results = {
                row["group"]: row
                for row in metric_rows
                if row["model"] == model_name
                and row["threshold"] == threshold
            }

            for field in [
                "records",
                "normal_records",
                "attack_records",
                "tn",
                "fp",
                "fn",
                "tp"
            ]:

                subgroup_total = (
                    results["overlapping"][field]
                    + results["non_overlapping"][field]
                )

                if results["all_test"][field] != subgroup_total:
                    raise ValueError(
                        f"Group reconciliation failed: "
                        f"{model_name}, {threshold}, {field}"
                    )

        with (
            output_dir / "model_metrics.json"
        ).open("w", encoding="utf-8") as file:

            json.dump(
                metric_rows,
                file,
                indent=4,
                allow_nan=False
            )

        manifest.update({
            "status": "complete",
            "finished_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "metric_rows": len(metric_rows),
            "group_reconciliation": "passed"
        })

        print("\n====================================")
        print("MODEL OVERLAP EVALUATION COMPLETED")
        print("====================================")
        print("Metric rows:", len(metric_rows))
        print("Group reconciliation: passed")
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

        if test_with_overlap is not None:
            test_with_overlap.unpersist()

        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()