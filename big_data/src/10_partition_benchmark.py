
from pathlib import Path
import csv
import time
import statistics

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    when
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "full"
RESULTS_DIR = PROJECT_DIR / "big_data" / "results"

PARTITION_COUNTS = [2, 4, 8, 16]
REPETITIONS = 3


def load_feature_names():

    features_file = (
        DATA_DIR / "NUSW-NB15_features.csv"
    )

    with open(
        features_file,
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
            f"Expected 49 columns, found {len(names)}"
        )

    return names


def main():

    spark = (
        SparkSession.builder
        .appName("IDS-Partition-Benchmark")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        feature_names = load_feature_names()

        files = [
            str(DATA_DIR / f"UNSW-NB15_{i}.csv")
            for i in range(1, 5)
        ]

        print("\nLoading full dataset...")

        df = spark.read.csv(
            files,
            header=False,
            inferSchema=False,
            mode="FAILFAST"
        )

        if len(df.columns) != len(feature_names):
            raise ValueError(
                "Dataset column count mismatch."
            )

        df = df.toDF(*feature_names)

        # Keep only columns needed for this benchmark.
        df = df.select(
            trim(col("proto")).alias("proto"),
            col("Label").cast("int").alias("label")
        )

        print("Total records:", df.count())

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        results_file = (
            RESULTS_DIR / "partition_benchmark.csv"
        )

        results = []

        print("\nStarting partition experiments...")

        for partitions in PARTITION_COUNTS:

            print(
                f"\nTesting {partitions} partitions..."
            )

            partitioned = df.repartition(
                partitions
            ).cache()

            # Materialize before timing aggregation.
            partitioned.count()

            times = []

            for repetition in range(
                1, REPETITIONS + 1
            ):

                start = time.perf_counter()

                grouped = (
                    partitioned
                    .groupBy("proto", "label")
                    .count()
                )

                # collect() executes the aggregation.
                result = grouped.collect()

                elapsed = (
                    time.perf_counter() - start
                )

                times.append(elapsed)

                print(
                    f"Run {repetition}: "
                    f"{elapsed:.3f} seconds"
                )

            average = statistics.mean(times)
            deviation = statistics.stdev(times)

            print(
                f"Average: {average:.3f} seconds"
            )

            results.append({
                "partitions": partitions,
                "records": 2540047,
                "run1": round(times[0], 3),
                "run2": round(times[1], 3),
                "run3": round(times[2], 3),
                "average_seconds": round(
                    average, 3
                ),
                "std_seconds": round(
                    deviation, 3
                )
            })

            partitioned.unpersist(
                blocking=True
            )

        with open(
            results_file,
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
                f"{result['average_seconds']:.3f}s"
            )

        print("\nResults saved to:", results_file)
        print("\nPARTITION BENCHMARK COMPLETED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
