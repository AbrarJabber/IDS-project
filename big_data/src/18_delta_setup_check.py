"""
Verify local Delta Lake installation.

Uses synthetic demonstration rows.
Checks writing, reading, appending, and reading an older version.
Creates a new output directory for each execution.
"""

import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, functions as F


def main():
    # This script lives in IDS-project/big_data/src/.
    project_root = Path(__file__).resolve().parents[2]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        project_root
        / "big_data"
        / "results"
        / "delta_setup"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    table_dir = output_dir / "demo_table"

    # A file URI works with Spark's filesystem interface on Windows.
    table_uri = table_dir.as_uri()

    print("Output folder:", output_dir)
    print("Starting Spark with Delta Lake support...")

    builder = (
        SparkSession.builder
        .appName("IDS_Delta_Setup_Check")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        # These are demonstration values, not actual network records.
        original = spark.range(0, 5).withColumn(
            "demo_label",
            (F.col("id") % 2).cast("int"),
        )

        print("\n1. Writing five demonstration rows...")

        original.write.format("delta").mode("errorifexists").save(
            table_uri
        )

        print("\n2. Reading and checking the saved values...")

        loaded = spark.read.format("delta").load(table_uri)
        actual_rows = [
            (row["id"], row["demo_label"])
            for row in loaded.orderBy("id").collect()
        ]
        expected_rows = [(i, i % 2) for i in range(5)]

        if actual_rows != expected_rows:
            raise RuntimeError("Saved data does not match the input.")

        loaded.orderBy("id").show()

        print("\n3. Appending two demonstration rows...")

        additional = spark.range(5, 7).withColumn(
            "demo_label",
            (F.col("id") % 2).cast("int"),
        )

        additional.write.format("delta").mode("append").save(
            table_uri
        )

        # Read again to obtain the latest table version.
        current = spark.read.format("delta").load(table_uri)

        current_rows = [
            (row["id"], row["demo_label"])
            for row in current.orderBy("id").collect()
        ]

        if current_rows != [(i, i % 2) for i in range(7)]:
            raise RuntimeError("Append verification failed.")

        print("\n4. Reading the original version...")

        original_version = (
            spark.read.format("delta")
            .option("versionAsOf", 0)
            .load(table_uri)
        )

        historical_rows = [
            (row["id"], row["demo_label"])
            for row in original_version.orderBy("id").collect()
        ]

        if historical_rows != expected_rows:
            raise RuntimeError("Historical version verification failed.")

        print("\n5. Displaying the table history...")

        delta_table = DeltaTable.forPath(spark, table_uri)
        delta_table.history().select(
            "version",
            "timestamp",
            "operation",
        ).orderBy("version").show(truncate=False)

        if not (table_dir / "_delta_log").is_dir():
            raise RuntimeError("Delta transaction log was not found.")

        summary = {
            "status": "passed",
            "purpose": "synthetic installation verification",
            "python_version": platform.python_version(),
            "spark_version": spark.version,
            "delta_spark_version": version("delta-spark"),
            "spark_master": spark.sparkContext.master,
            "initial_rows": len(actual_rows),
            "current_rows": len(current_rows),
            "version_0_rows": len(historical_rows),
            "initial_values_match": actual_rows == expected_rows,
            "historical_values_match": historical_rows == expected_rows,
            "table_path": str(table_dir),
        }

        summary_path = output_dir / "setup_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\nDELTA SETUP CHECK PASSED")
        print("Initial rows:", len(actual_rows))
        print("Current rows:", len(current_rows))
        print("Version 0 rows:", len(historical_rows))
        print("Summary saved to:", summary_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()