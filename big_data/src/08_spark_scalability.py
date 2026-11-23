
from pathlib import Path
import argparse
import csv
import subprocess
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_DIR / "data" / "raw"
    / "UNSW_NB15_training-set.csv"
)

RESULTS_DIR = PROJECT_DIR / "big_data" / "results"
RESULTS_FILE = RESULTS_DIR / "scalability_results.csv"

DATA_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
WORKER_COUNTS = [1, 2, 4]
REPETITIONS = 3

FIELDS = [
    "workers",
    "fraction",
    "repetition",
    "records",
    "load_seconds",
    "preprocess_seconds",
    "training_seconds",
    "total_seconds"
]


def create_pipeline(columns):

    categorical_columns = [
        "proto",
        "service",
        "state"
    ]

    numeric_columns = [
        c for c in columns
        if c not in categorical_columns
        and c not in ["label", "attack_cat"]
    ]

    stages = []

    for name in categorical_columns:
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
                for name in categorical_columns
            ],
            outputCols=[
                name + "_encoded"
                for name in categorical_columns
            ],
            handleInvalid="keep"
        )
    )

    stages.append(
        VectorAssembler(
            inputCols=numeric_columns + [
                name + "_encoded"
                for name in categorical_columns
            ],
            outputCol="features"
        )
    )

    stages.append(
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
    )

    return Pipeline(stages=stages)


def run_experiment(workers, fraction, repetition):

    spark = (
        SparkSession.builder
        .appName(
            f"IDS-Scalability-{workers}-{fraction}"
        )
        .master(f"local[{workers}]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        total_start = time.perf_counter()

        print(
            f"\nWorkers={workers}, "
            f"Fraction={fraction}, "
            f"Repetition={repetition}",
            flush=True
        )

        load_start = time.perf_counter()

        df = spark.read.csv(
            str(DATA_FILE),
            header=True,
            inferSchema=True
        )

        df = df.drop("id")

        if fraction < 1.0:
            df = df.sample(
                withReplacement=False,
                fraction=fraction,
                seed=42
            )

        df = df.cache()
        records = df.count()

        load_seconds = (
            time.perf_counter() - load_start
        )

        preprocess_start = time.perf_counter()

        processed = df.withColumn(
            "service",
            when(
                col("service") == "-",
                "unknown"
            ).otherwise(col("service"))
        )

        processed = processed.cache()
        processed.count()

        preprocess_seconds = (
            time.perf_counter() - preprocess_start
        )

        training_start = time.perf_counter()

        pipeline = create_pipeline(
            processed.columns
        )

        model = pipeline.fit(processed)

        training_seconds = (
            time.perf_counter() - training_start
        )

        total_seconds = (
            time.perf_counter() - total_start
        )

        print(
            f"Records: {records}, "
            f"Training: {training_seconds:.2f}s, "
            f"Total: {total_seconds:.2f}s",
            flush=True
        )

        return {
            "workers": workers,
            "fraction": fraction,
            "repetition": repetition,
            "records": records,
            "load_seconds": round(
                load_seconds, 3
            ),
            "preprocess_seconds": round(
                preprocess_seconds, 3
            ),
            "training_seconds": round(
                training_seconds, 3
            ),
            "total_seconds": round(
                total_seconds, 3
            )
        }

    finally:
        spark.stop()


def save_result(result):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = RESULTS_FILE.exists()

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def run_all():

    experiments = [
        (workers, fraction, repetition)
        for workers in WORKER_COUNTS
        for fraction in DATA_FRACTIONS
        for repetition in range(
            1, REPETITIONS + 1
        )
    ]

    print(
        f"\nStarting {len(experiments)} "
        "scalability experiments..."
    )

    for index, (
        workers,
        fraction,
        repetition
    ) in enumerate(experiments, 1):

        print(
            f"\nExperiment {index}/"
            f"{len(experiments)}"
        )

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--workers",
            str(workers),
            "--fraction",
            str(fraction),
            "--repetition",
            str(repetition)
        ]

        result = subprocess.run(
            command,
            cwd=str(PROJECT_DIR)
        )

        if result.returncode != 0:
            print(
                "Experiment failed. "
                "Stopping to avoid incomplete results."
            )
            sys.exit(result.returncode)

    print("\nALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--single",
        action="store_true"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--repetition",
        type=int,
        default=1
    )

    args = parser.parse_args()

    if args.single:

        result = run_experiment(
            args.workers,
            args.fraction,
            args.repetition
        )

        save_result(result)

    else:
        run_all()
