"""
Convert the official UNSW-NB15 splits to Parquet and Delta.

Verify column names, data types, row counts, and complete row equality,
including duplicate multiplicities. No model training is performed.
"""

import json
import os
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def configure_windows():
    """Use the Hadoop utilities installed during the setup check."""
    if os.name != "nt":
        return

    hadoop_dir = Path(
        os.environ.get("HADOOP_HOME")
        or str(Path.home() / "hadoop-3.3.4")
    )

    for filename in ("winutils.exe", "hadoop.dll"):
        required = hadoop_dir / "bin" / filename
        if not required.is_file():
            raise FileNotFoundError(f"Missing Windows utility: {required}")

    os.environ["HADOOP_HOME"] = str(hadoop_dir)
    bin_path = str(hadoop_dir / "bin")
    existing_path = os.environ.get("PATH", "")

    if bin_path.lower() not in [
        part.lower() for part in existing_path.split(os.pathsep)
    ]:
        os.environ["PATH"] = bin_path + os.pathsep + existing_path


def column_types(df):
    # Compare names and data types; reader nullability metadata may differ.
    return [
        (field.name, field.dataType.simpleString())
        for field in df.schema.fields
    ]


def main():
    configure_windows()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    root = Path(__file__).resolve().parents[2]

    inputs = {
        "train": root / "data/raw/UNSW_NB15_training-set.csv",
        "test": root / "data/raw/UNSW_NB15_testing-set.csv",
    }

    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    storage_dir = root / "big_data/storage" / run_id
    results_dir = root / "big_data/results/storage_conversion" / run_id

    storage_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "status": "running",
        "run_id": run_id,
        "python": platform.python_version(),
        "pyspark": version("pyspark"),
        "delta_spark": version("delta-spark"),
        "storage_directory": str(storage_dir),
        "verification": (
            "Column names/types, counts, and bidirectional exceptAll. "
            "Duplicate multiplicities are preserved."
        ),
        "datasets": [],
    }

    builder = (
        SparkSession.builder
        .appName("IDS_Official_Split_Storage_Conversion")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = None

    try:
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        # Match the loading approach used in the earlier EDA script.
        train = spark.read.csv(
            inputs["train"].as_uri(),
            header=True,
            inferSchema=True,
            mode="FAILFAST",
        )

        test = spark.read.csv(
            inputs["test"].as_uri(),
            header=True,
            schema=train.schema,
            mode="FAILFAST",
        )

        expected_counts = {"train": 175341, "test": 82332}

        for name, source in (("train", train), ("test", test)):
            print(f"\n=== CONVERTING {name.upper()} ===", flush=True)

            source = source.cache()

            try:
                source_count = source.count()

                if source_count != expected_counts[name]:
                    raise ValueError(
                        f"{name}: expected {expected_counts[name]:,} rows "
                        f"from the previous audit, found {source_count:,}. "
                        "Check the input file before continuing."
                    )

                schema_path = results_dir / f"{name}_schema.json"
                schema_path.write_text(
                    source.schema.json(),
                    encoding="utf-8",
                )

                class_counts = [
                    row.asDict()
                    for row in source.groupBy("label")
                    .count()
                    .orderBy("label")
                    .collect()
                ]

                entry = {
                    "dataset": name,
                    "source": str(inputs[name]),
                    "source_bytes": inputs[name].stat().st_size,
                    "records": source_count,
                    "columns": len(source.columns),
                    "class_counts": class_counts,
                    "formats": [],
                }
                manifest["datasets"].append(entry)

                for fmt in ("parquet", "delta"):
                    target = storage_dir / name / fmt
                    target_uri = target.as_uri()

                    print(f"Writing {fmt}...", flush=True)

                    (
                        source.repartition(4)
                        .write.format(fmt)
                        .option("compression", "snappy")
                        .mode("errorifexists")
                        .save(target_uri)
                    )

                    restored = (
                        spark.read.format(fmt).load(target_uri)
                    )

                    if column_types(source) != column_types(restored):
                        raise RuntimeError(
                            f"{name}/{fmt}: column names or types changed."
                        )

                    # Explicitly use the source column order.
                    restored = restored.select(*source.columns)
                    restored_count = restored.count()

                    # Check all values and duplicate occurrences.
                    # Both comparisons execute as distributed Spark jobs.
                    missing_rows = source.exceptAll(restored).count()
                    extra_rows = restored.exceptAll(source).count()

                    passed = (
                        restored_count == source_count
                        and missing_rows == 0
                        and extra_rows == 0
                    )

                    entry["formats"].append({
                        "format": fmt,
                        "path": str(target),
                        "records": restored_count,
                        "missing_rows": missing_rows,
                        "extra_rows": extra_rows,
                        "verified": passed,
                    })

                    if not passed:
                        raise RuntimeError(
                            f"{name}/{fmt}: conversion verification failed."
                        )

                    print(
                        f"PASS: {name} / {fmt} | "
                        f"rows={restored_count:,} | "
                        f"missing={missing_rows} | extra={extra_rows}",
                        flush=True,
                    )

            finally:
                source.unpersist()

        manifest["status"] = "passed"

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise

    finally:
        manifest_path = results_dir / "conversion_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print("\nManifest:", manifest_path, flush=True)

        if spark is not None:
            spark.stop()

    print("\nOFFICIAL SPLIT CONVERSION PASSED")
    print("Storage folder:", storage_dir)
    print("Verification results:", results_dir)


if __name__ == "__main__":
    main()
    