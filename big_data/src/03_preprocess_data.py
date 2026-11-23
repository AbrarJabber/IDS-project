
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

PROJECT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Preprocessing")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def prepare_data(df):

    # Remove record identifier
    df = df.drop("id")

    # Replace unidentified service with a clear category
    df = df.withColumn(
        "service",
        when(
            col("service") == "-",
            "unknown"
        ).otherwise(col("service"))
    )

    return df


try:
    print("\nLoading datasets...")

    train_df = spark.read.csv(
        str(RAW_DIR / "UNSW_NB15_training-set.csv"),
        header=True,
        inferSchema=True
    )

    test_df = spark.read.csv(
        str(RAW_DIR / "UNSW_NB15_testing-set.csv"),
        header=True,
        inferSchema=True
    )

    print("\nPreprocessing training dataset...")
    train_clean = prepare_data(train_df)

    print("\nPreprocessing testing dataset...")
    test_clean = prepare_data(test_df)

    print("\nSaving processed datasets...")



    print("\n=== PREPROCESSING RESULTS ===")
    print("Training rows:", train_clean.count())
    print("Testing rows:", test_clean.count())
    print("Columns:", len(train_clean.columns))

    print("\nUpdated service values:")
    train_clean.groupBy("service").count().orderBy(
        col("count").desc()
    ).show(10, truncate=False)

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY")

finally:
    spark.stop()
