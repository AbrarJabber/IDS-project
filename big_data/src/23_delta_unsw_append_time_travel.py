"""Demonstrate Delta Lake append and time travel using REAL UNSW-NB15 records.

Place this file in IDS-project/big_data/src/ alongside scripts 18–22.

Input:  data/raw/UNSW_NB15_training-set.csv (the official training split)
Output: big_data/results/delta_real_data/<timestamp>/
        big_data/storage_delta_history/<timestamp>/unsw_history_demo/

Demonstration (defaults):
  Version 0: first 1,000 records ordered by the official id.
  Version 1: append the next 250 distinct records from the same source.
  Time travel: read version 0 again *after* the append and verify its contents.

Only new, timestamped output folders are created; the input CSV and existing
Delta/Parquet tables are never modified. This is a correctness demonstration,
not a performance benchmark or a production ingestion system.
"""

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def configure_windows():
    """Match the working Windows/Hadoop setup in scripts 19–21."""
    if os.name != "nt":
        return

    hadoop_dir = Path(
        os.environ.get("HADOOP_HOME")
        or str(Path.home() / "hadoop-3.3.4")
    )
    for executable in ("winutils.exe", "hadoop.dll"):
        required = hadoop_dir / "bin" / executable
        if not required.is_file():
            raise FileNotFoundError(f"Missing Windows Hadoop utility: {required}")

    os.environ["HADOOP_HOME"] = str(hadoop_dir)
    bin_path = str(hadoop_dir / "bin")
    existing = os.environ.get("PATH", "")
    if bin_path.lower() not in [
        part.lower() for part in existing.split(os.pathsep)
    ]:
        os.environ["PATH"] = bin_path + os.pathsep + existing


def write_csv(path, records, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def require_same_rows(expected, observed, label):
    """Compare every value and duplicate multiplicity, in both directions."""
    if [(c.name, c.dataType.simpleString()) for c in expected.schema] != [
        (c.name, c.dataType.simpleString()) for c in observed.schema
    ]:
        raise AssertionError(f"{label}: column names or types changed")

    expected_columns = expected.columns
    observed = observed.select(*expected_columns)
    missing = expected.exceptAll(observed).count()
    extra = observed.exceptAll(expected).count()
    if missing or extra:
        raise AssertionError(
            f"{label}: rows differ (missing={missing}, extra={extra})"
        )
    return {"missing_rows": missing, "extra_rows": extra, "passed": True}


def snapshot_summary(dataframe, name, table_version):
    from pyspark.sql import functions as F

    row = dataframe.agg(
        F.count("*").alias("records"),
        F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).alias("normal"),
        F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("attack"),
        F.sum(F.col("sbytes").cast("long")).alias("total_source_bytes"),
        F.sum(F.col("dbytes").cast("long")).alias("total_destination_bytes"),
    ).first()
    return {
        "snapshot": name,
        "version": table_version,
        "records": int(row["records"]),
        "normal": int(row["normal"]),
        "attack": int(row["attack"]),
        "total_source_bytes": int(row["total_source_bytes"] or 0),
        "total_destination_bytes": int(row["total_destination_bytes"] or 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root (normally inferred from big_data/src/).",
    )
    parser.add_argument("--initial-rows", type=int, default=1000)
    parser.add_argument("--append-rows", type=int, default=250)
    args = parser.parse_args()

    if args.initial_rows < 1 or args.append_rows < 1:
        parser.error("--initial-rows and --append-rows must both be positive")

    configure_windows()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    # Imports are delayed so --help works even outside the project environment.
    from delta import configure_spark_with_delta_pip
    from delta.tables import DeltaTable
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    root = args.project_dir.resolve()
    source_csv = root / "data" / "raw" / "UNSW_NB15_training-set.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"Official UNSW-NB15 training file not found: {source_csv}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results = root / "big_data" / "results" / "delta_real_data" / run_id
    table_dir = (
        root / "big_data" / "storage_delta_history" / run_id / "unsw_history_demo"
    )
    results.mkdir(parents=True, exist_ok=False)
    table_uri = table_dir.as_uri()

    manifest = {
        "status": "running",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_csv),
        "source_bytes": source_csv.stat().st_size,
        "source_role": "official training split (not the full original collection)",
        "table_path": str(table_dir),
        "table_uri": table_uri,
        "results_directory": str(results),
        "python_version": platform.python_version(),
        "pyspark_version": version("pyspark"),
        "delta_spark_version": version("delta-spark"),
        "initial_rows_requested": args.initial_rows,
        "append_rows_requested": args.append_rows,
        "method": (
            "Select the first N original records by ascending id; write "
            "the first batch to Delta version 0 and append the second "
            "batch as version 1; check every value using bidirectional "
            "exceptAll and re-read version 0 after the append."
        ),
        "cautions": [
            "The actual UNSW-NB15 records come from the official training CSV.",
            "No fabricated records, updates, overwrites or deletions.",
            "Single local Spark session; no multi-node scalability claim.",
            "This is not a write-performance or ingest-throughput benchmark.",
            "Historical reads require previous data files to be retained; "
            "do not VACUUM this table before inspecting old versions.",
        ],
    }
    (results / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    spark = None
    sample = None
    try:
        builder = (
            SparkSession.builder
            .appName("IDS_Real_UNSW_Delta_Append_Time_Travel")
            .master("local[2]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "4")
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
        manifest["spark_version"] = spark.version
        manifest["spark_master"] = spark.sparkContext.master

        print("\n1. Reading the official UNSW-NB15 training CSV...", flush=True)
        source = spark.read.csv(
            source_csv.as_uri(),
            header=True,
            inferSchema=True,
            mode="FAILFAST",
        )
        required = {"id", "label", "sbytes", "dbytes"}
        if not required.issubset(source.columns):
            raise ValueError(f"Required columns missing: {required - set(source.columns)}")

        source_count = source.count()
        if source_count != 175341:
            raise ValueError(
                f"Unexpected training size: {source_count}. Expected 175341."
            )
        manifest["source_records"] = source_count
        manifest["source_columns"] = len(source.columns)
        total_demo = args.initial_rows + args.append_rows
        if total_demo > source_count:
            raise ValueError("Requested sample is larger than the source data")

        # Two disjoint, deterministic batches made only from original rows.
        # Order by official record ID, then generate a row number within sample.
        sample = (
            source.orderBy(F.col("id").asc())
            .limit(total_demo)
            .withColumn("_demo_sequence", F.row_number().over(Window.orderBy("id")))
            .cache()
        )
        if sample.count() != total_demo:
            raise AssertionError("Incorrect demonstration sample count")
        if sample.filter(F.col("id").isNull()).count():
            raise ValueError("Some selected rows have null IDs")
        if sample.select("id").distinct().count() != total_demo:
            raise ValueError("Selected official record IDs are not unique")

        first_batch = sample.filter(
            F.col("_demo_sequence") <= args.initial_rows
        ).drop("_demo_sequence")
        appended_batch = sample.filter(
            F.col("_demo_sequence") > args.initial_rows
        ).drop("_demo_sequence")
        whole_sample = sample.drop("_demo_sequence")

        initial_expected = snapshot_summary(first_batch, "source_initial_batch", None)
        append_expected = snapshot_summary(appended_batch, "source_appended_batch", None)
        if initial_expected["records"] != args.initial_rows:
            raise AssertionError("Incorrect initial batch count")
        if append_expected["records"] != args.append_rows:
            raise AssertionError("Incorrect append batch count")

        print("\n2. Creating Delta table, version 0...", flush=True)
        (
            first_batch.repartition(2)
            .write.format("delta")
            .option("compression", "snappy")
            .mode("errorifexists")
            .save(table_uri)
        )
        version_zero_before = (
            spark.read.format("delta")
            .option("versionAsOf", 0)
            .load(table_uri)
        )
        zero_check = require_same_rows(
            first_batch, version_zero_before, "version 0 immediately after creation"
        )
        v0 = snapshot_summary(version_zero_before, "version_0_initial", 0)

        print("\n3. Appending the next genuine training records...", flush=True)
        (
            appended_batch.repartition(2)
            .write.format("delta")
            .option("compression", "snappy")
            .mode("append")
            .save(table_uri)
        )

        # Force an explicitly versioned read to avoid ambiguous latest-snapshot timing.
        version_one = (
            spark.read.format("delta")
            .option("versionAsOf", 1)
            .load(table_uri)
        )
        one_check = require_same_rows(
            whole_sample, version_one, "version 1 after append"
        )
        v1 = snapshot_summary(version_one, "version_1_after_append", 1)
        if v1["records"] != args.initial_rows + args.append_rows:
            raise AssertionError("Delta append did not produce the expected count")

        print("\n4. Reading historical Delta version 0 AFTER append...", flush=True)
        historical = (
            spark.read.format("delta")
            .option("versionAsOf", 0)
            .load(table_uri)
        )
        historical_check = require_same_rows(
            first_batch, historical, "version 0 recovered after append"
        )
        historical_summary = snapshot_summary(
            historical, "version_0_recovered_after_append", 0
        )
        if historical_summary["records"] != v0["records"]:
            raise AssertionError("Historical version 0 count changed")

        print("\n5. Checking the Delta transaction log and history...", flush=True)
        if not (table_dir / "_delta_log").is_dir():
            raise AssertionError("Missing Delta transaction-log directory")
        history = [
            {
                "version": int(r["version"]),
                "timestamp": r["timestamp"].isoformat() if r["timestamp"] else "",
                "operation": str(r["operation"]),
            }
            for r in (
                DeltaTable.forPath(spark, table_uri)
                .history()
                .select("version", "timestamp", "operation")
                .orderBy("version")
                .collect()
            )
        ]
        if [r["version"] for r in history] != [0, 1]:
            raise AssertionError(f"Unexpected version history: {history}")
        if any(r["operation"] != "WRITE" for r in history):
            raise AssertionError(f"Unexpected recorded write operation: {history}")

        snapshots = [v0, v1, historical_summary]
        write_csv(
            results / "snapshot_summary.csv",
            snapshots,
            [
                "snapshot", "version", "records", "normal", "attack",
                "total_source_bytes", "total_destination_bytes",
            ],
        )
        write_csv(
            results / "delta_history.csv",
            history,
            ["version", "timestamp", "operation"],
        )
        verification = {
            "version_0_creation": zero_check,
            "version_1_after_append": one_check,
            "version_0_time_travel_after_append": historical_check,
            "version_numbers": [r["version"] for r in history],
            "transaction_log_found": True,
            "real_records_not_synthetic": True,
            "status": "passed",
        }
        (results / "verification.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )

        manifest.update({
            "status": "passed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "observed_snapshots": snapshots,
            "history": history,
            "verification": verification,
        })
        (results / "report_ready_summary.md").write_text(
            "# Delta Lake append and historical snapshot experiment\n\n"
            "The official UNSW-NB15 training dataset (175,341 records) "
            "supplied genuine network records for a small derived Delta table. "
            f"Version 0 contained **{v0['records']:,} rows** "
            f"({v0['normal']:,} normal; {v0['attack']:,} attack). "
            f"A disjoint batch of **{append_expected['records']:,} real records** "
            "was appended without modifying or overwriting the original source. "
            f"Version 1 contained **{v1['records']:,} rows** "
            f"({v1['normal']:,} normal; {v1['attack']:,} attack). "
            "Reading **version 0 after the append** returned exactly the original "
            f"**{historical_summary['records']:,} records**, confirming historical "
            "snapshot access. Schema and bidirectional `exceptAll()` comparisons "
            "confirmed every value and duplicate occurrence for both versions. "
            "The Delta transaction log recorded versions 0 and 1.\n\n"
            "**Interpretation:** The experiment demonstrates append operations "
            "and version-based historical reads on genuine UNSW-NB15 rows, "
            "complementing the separate CSV–Parquet–Delta query benchmark. "
            "It does not measure update throughput, concurrent transactions, "
            "or multi-node execution. Historical access depends on retaining "
            "the underlying data files.\n",
            encoding="utf-8",
        )
        print("\nDELTA REAL-DATA HISTORY EXPERIMENT PASSED")
        print(f"Version 0: {v0['records']:,} rows")
        print(f"Version 1: {v1['records']:,} rows")
        print(f"Recovered version 0: {historical_summary['records']:,} rows")
        print("Bidirectional data-equality checks: PASSED")
        print("Recorded Delta table versions: 0 and 1")
        print("Results:", results)
        print("Delta table:", table_dir)

    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        (results / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if sample is not None:
            sample.unpersist()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
