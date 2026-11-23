
from pathlib import Path
import csv
import time
import statistics

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    lower,
    when,
    count,
    avg
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "full"
RESULTS_DIR = PROJECT_DIR / "big_data" / "results"

PARTITION_COUNTS = [2, 8, 16, 4]
REPETITIONS = 3


def load_feature_names():

    file_path = (
        DATA_DIR / "NUSW-NB15_features.csv"
    )

    with open(
        file_path,
        "r",
        encoding="cp1252",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        names = [
            row["Name"].strip().replace(" ", "")
            for row in reader
        ]

    if len(names) != 49:
        raise ValueError(
            "Expected 49 feature names."
        )

    return names


def run_pipeline(spark, feature_names, partitions):

    files = [
        str(DATA_DIR / f"UNSW-NB15_{i}.csv")
        for i in range(1, 5)
    ]

    start = time.perf_counter()

    df = spark.read.csv(
        files,
        header=False,
        inferSchema=False,
        mode="FAILFAST"
    )

    if len(df.columns) != len(feature_names):
        raise ValueError(
            "CSV column count mismatch."
        )

    df = df.toDF(*feature_names)

    df = df.select(
        trim(col("proto")).alias("proto"),
        col("Label").cast("int").alias("label"),
        col("sbytes").cast("double").alias("sbytes"),
        col("dbytes").cast("double").alias("dbytes"),
        lower(
            trim(col("attack_cat"))
        ).alias("attack_cat")
    )

    df = df.withColumn(
        "attack_cat",
        when(
            col("label") == 0,
            "normal"
        ).when(
            col("attack_cat") == "backdoors",
            "backdoor"
        ).otherwise(
            col("attack_cat")
        )
    )

    df = df.repartition(partitions)

    grouped = (
        df.groupBy("proto", "attack_cat")
        .agg(
            count("*").alias("records"),
            avg("sbytes").alias("avg_source_bytes"),
            avg("dbytes").alias("avg_destination_bytes")
        )
    )

    result = grouped.collect()

    elapsed = time.perf_counter() - start

    total_records = sum(
        row["records"] for row in result
    )

    if total_records != 2540047:
        raise ValueError(
            f"Unexpected record count: {total_records}"
        )

    return elapsed, len(result)


def main():

    spark = (
        SparkSession.builder
        .appName("IDS-End-to-End-Benchmark")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        feature_names = load_feature_names()

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        results = []

        print("\nStarting end-to-end benchmark...")

        for partitions in PARTITION_COUNTS:

            print(
                f"\nTesting {partitions} partitions"
            )

            print("Warm-up run...")

            warmup_time, _ = run_pipeline(
                spark,
                feature_names,
                partitions
            )

            print(
                f"Warm-up: {warmup_time:.2f}s"
            )

            times = []

            for repetition in range(
                1, REPETITIONS + 1
            ):

                elapsed, groups = run_pipeline(
                    spark,
                    feature_names,
                    partitions
                )

                times.append(elapsed)

                print(
                    f"Run {repetition}: "
                    f"{elapsed:.2f}s, "
                    f"groups={groups}"
                )

            average_time = statistics.mean(times)
            median_time = statistics.median(times)
            std_time = statistics.stdev(times)

            results.append({
                "partitions": partitions,
                "run1": round(times[0], 3),
                "run2": round(times[1], 3),
                "run3": round(times[2], 3),
                "average_seconds": round(
                    average_time, 3
                ),
                "median_seconds": round(
                    median_time, 3
                ),
                "std_seconds": round(
                    std_time, 3
                )
            })

            print(
                f"Average: {average_time:.2f}s"
            )

            print(
                f"Median: {median_time:.2f}s"
            )

        output_file = (
            RESULTS_DIR / "end_to_end_benchmark.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys()
            )

            writer.writeheader()
            writer.writerows(results)

        print("\n=== FINAL RESULTS ===")

        for result in results:

            print(
                f"{result['partitions']} partitions: "
                f"average={result['average_seconds']:.3f}s, "
                f"median={result['median_seconds']:.3f}s"
            )

        print("\nResults saved to:", output_file)
        print("\nEND-TO-END BENCHMARK COMPLETED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
