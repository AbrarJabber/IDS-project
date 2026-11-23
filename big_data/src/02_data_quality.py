
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Data-Quality")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    train_df = spark.read.csv(
        str(DATA_DIR / "UNSW_NB15_training-set.csv"),
        header=True,
        inferSchema=True
    )

    test_df = spark.read.csv(
        str(DATA_DIR / "UNSW_NB15_testing-set.csv"),
        header=True,
        inferSchema=True
    )

    for name, df in [
        ("TRAINING", train_df),
        ("TESTING", test_df)
    ]:
        print(f"\n=== {name} DATA QUALITY ===")

        total = df.count()
        print("Total records:", total)

        # Exclude row identifier when checking duplicates
        feature_columns = [
            c for c in df.columns if c != "id"
        ]

        unique_count = df.dropDuplicates(
            feature_columns
        ).count()

        print("Duplicate records:", total - unique_count)

        # Check missing values and NaN values
        numeric_types = {
            "integer", "long", "double", "float"
        }

        missing_expressions = []

        for field in df.schema.fields:
            column_name = field.name

            condition = col(column_name).isNull()

            if field.dataType.simpleString() in numeric_types:
                condition = condition | isnan(col(column_name))

            missing_expressions.append(
                count(
                    when(condition, 1)
                ).alias(column_name)
            )

        missing_row = df.select(
            missing_expressions
        ).first()

        missing = {
            column: value
            for column, value in missing_row.asDict().items()
            if value > 0
        }

        print("Columns with missing/NaN values:", missing)

        # Check class distribution
        print("\nAttack categories:")
        df.groupBy("attack_cat").count().orderBy(
            col("count").desc()
        ).show(15, truncate=False)

        # Inspect categorical placeholders
        print("\nService values:")
        df.groupBy("service").count().orderBy(
            col("count").desc()
        ).show(15, truncate=False)

        # Validate binary labels
        print("\nBinary labels:")
        df.groupBy("label").count().show()

        # Check for invalid negative values
        for column_name in [
            "dur", "spkts", "dpkts",
            "sbytes", "dbytes"
        ]:
            invalid_count = df.filter(
                col(column_name) < 0
            ).count()

            print(
                f"Negative {column_name}:",
                invalid_count
            )

    print("\nDATA QUALITY ANALYSIS COMPLETED")

finally:
    spark.stop()
