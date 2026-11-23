
from pathlib import Path
import csv
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
    when,
    trim,
    lower
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "full"

FEATURES_FILE = DATA_DIR / "NUSW-NB15_features.csv"

DATA_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv"
]


def load_feature_names():

    with open(
    FEATURES_FILE,
    "r",
    encoding="cp1252",
    newline=""
) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    print("\nFeature description columns:")
    print(reader.fieldnames)

    if not rows:
        raise ValueError(
            "The feature description file is empty."
        )

    if "Name" not in reader.fieldnames:
        raise ValueError(
            "Expected a 'Name' column in the "
            "feature description file."
        )

    feature_names = [
        row["Name"].strip()
        for row in rows
    ]

    if any(not name for name in feature_names):
        raise ValueError(
            "Some feature names are empty."
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Duplicate feature names were found."
        )

    return feature_names


def main():

    print("Starting full UNSW-NB15 analysis...")

    for file in DATA_FILES + [FEATURES_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Missing file: {file}"
            )

    feature_names = load_feature_names()

    print("\nFeature count:", len(feature_names))
    print("First five features:", feature_names[:5])
    print("Last five features:", feature_names[-5:])

    spark = (
        SparkSession.builder
        .appName("UNSW-NB15-Large-Scale-Analysis")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        start_time = time.perf_counter()

        print("\nLoading four CSV files...")

        # Original UNSW-NB15 files do not have headers.
        # Use the feature-description file for column names.

        df = spark.read.csv(
            [str(file) for file in DATA_FILES],
            header=False,
            inferSchema=False,
            mode="FAILFAST"
        )

        actual_columns = len(df.columns)

        print("CSV column count:", actual_columns)

        if actual_columns != len(feature_names):
            raise ValueError(
                f"Column mismatch: CSV has "
                f"{actual_columns} columns, "
                f"but feature file has "
                f"{len(feature_names)} names."
            )

        df = df.toDF(*feature_names)

        df = df.cache()

        total_records = df.count()

        loading_time = (
            time.perf_counter() - start_time
        )

        print("\n=== DATASET INFORMATION ===")
        print("Total records:", total_records)
        print("Total columns:", len(df.columns))
        print("Spark partitions:", df.rdd.getNumPartitions())
        print(
            f"Loading time: {loading_time:.2f} seconds"
        )

        print("\n=== DATASET SCHEMA ===")
        df.printSchema()

        print("\n=== FIRST FIVE RECORDS ===")
        df.select(
            *feature_names[:5]
        ).show(5, truncate=False)

        print("\n=== BINARY LABEL DISTRIBUTION ===")

        if "Label" in df.columns:
            label_column = "Label"
        elif "label" in df.columns:
            label_column = "label"
        else:
            raise ValueError(
                "Label column not found."
            )

        df.groupBy(label_column).count().orderBy(
            label_column
        ).show(truncate=False)

        print("\n=== ATTACK CATEGORY DISTRIBUTION ===")

        if "attack_cat" not in df.columns:
            raise ValueError(
                "attack_cat column not found."
            )

        df.groupBy("attack_cat").count().orderBy(
            col("count").desc()
        ).show(30, truncate=False)

        print("\n=== EMPTY VALUE CHECK ===")

        empty_counts = df.select([
            spark_sum(
                when(
                    col(name).isNull()
                    | (trim(col(name)) == ""),
                    1
                ).otherwise(0)
            ).alias(name)
            for name in df.columns
        ])

        empty_counts.show(
            truncate=False,
            vertical=True
        )

        print("\n=== NORMALIZED ATTACK CATEGORIES ===")

        normalized = df.withColumn(
            "attack_category_clean",
            lower(trim(col("attack_cat")))
        )

        normalized.groupBy(
            "attack_category_clean"
        ).count().orderBy(
            col("count").desc()
        ).show(30, truncate=False)

        print("\nFULL DATASET ANALYSIS COMPLETED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
