"""
Convert the four original UNSW-NB15 files to Parquet and Delta.

Preserve the raw Spark-loaded values and duplicate occurrences.
Save verification results, schema, and storage measurements.
"""

import csv
import json
import os
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from delta import configure_spark_with_delta_pip
from pyspark import StorageLevel
from pyspark.sql import SparkSession


def configure_windows():
    if os.name != "nt":
        return

    directory = Path(
        os.environ.get("HADOOP_HOME")
        or str(Path.home() / "hadoop-3.3.4")
    )

    for filename in ("winutils.exe", "hadoop.dll"):
        path = directory / "bin" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Windows utility: {path}")

    os.environ["HADOOP_HOME"] = str(directory)
    bin_path = str(directory / "bin")
    current_path = os.environ.get("PATH", "")

    if bin_path.lower() not in [
        item.lower() for item in current_path.split(os.pathsep)
    ]:
        os.environ["PATH"] = bin_path + os.pathsep + current_path


def column_types(df):
    return [
        (field.name, field.dataType.simpleString())
        for field in df.schema.fields
    ]


def measure_storage(directory):
    """Separate table data, Delta metadata, and auxiliary files."""
    data_bytes = 0
    metadata_bytes = 0
    auxiliary_bytes = 0

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        size = path.stat().st_size
        relative = path.relative_to(directory)

        if path.name.startswith("."):
            auxiliary_bytes += size
        elif "_delta_log" in relative.parts:
            metadata_bytes += size
        elif path.suffix == ".parquet":
            data_bytes += size
        else:
            auxiliary_bytes += size

    return {
        "data_bytes": data_bytes,
        "metadata_bytes": metadata_bytes,
        "auxiliary_bytes": auxiliary_bytes,
        "total_bytes": data_bytes + metadata_bytes + auxiliary_bytes,
    }


def main():
    configure_windows()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    root = Path(__file__).resolve().parents[2]
    full_dir = root / "data/full"

    csv_paths = [
        full_dir / f"UNSW-NB15_{number}.csv"
        for number in range(1, 5)
    ]
    features_path = full_dir / "NUSW-NB15_features.csv"

    for path in csv_paths + [features_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    # Match the column naming used by the earlier EDA loader.
    with features_path.open(
        encoding="cp1252", newline=""
    ) as stream:
        names = [
            row["Name"].strip().replace(" ", "")
            for row in csv.DictReader(stream)
        ]

    names = [
        "label" if name.lower() == "label" else name
        for name in names
    ]

    if len(names) != 49 or len(set(n.lower() for n in names)) != 49:
        raise ValueError("Expected 49 unique full-dataset column names.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    storage_dir = root / "big_data/storage_full" / run_id
    results_dir = root / "big_data/results/full_storage_conversion" / run_id

    storage_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)

    input_bytes = sum(path.stat().st_size for path in csv_paths)

    manifest = {
        "status": "running",
        "run_id": run_id,
        "python": platform.python_version(),
        "pyspark": version("pyspark"),
        "delta_spark": version("delta-spark"),
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in csv_paths
        ],
        "features_file": str(features_path),
        "storage_directory": str(storage_dir),
        "notes": [
            "All 49 columns retain their initial string representation.",
            "No deduplication, label correction, or imputation is applied.",
            "Verification includes duplicate multiplicities.",
            "Source persistence is DISK_ONLY for conversion verification.",
            "Write timings are single observations, not benchmark results.",
        ],
        "formats": [],
    }

    storage_rows = []
    spark = None
    source = None

    builder = (
        SparkSession.builder
        .appName("IDS_Full_Dataset_Storage_Conversion")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    try:
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        print("\nLoading the four original CSV files...", flush=True)

        raw = spark.read.csv(
            [path.as_uri() for path in csv_paths],
            header=False,
            inferSchema=False,
            mode="FAILFAST",
        )

        if len(raw.columns) != 49:
            raise ValueError(
                f"Expected 49 CSV columns, found {len(raw.columns)}."
            )

        source = raw.toDF(*names).persist(StorageLevel.DISK_ONLY)
        source_count = source.count()

        print(f"Loaded records: {source_count:,}", flush=True)
        print(f"Columns: {len(source.columns)}", flush=True)

        if source_count == 0:
            raise ValueError("The full dataset is empty.")

        previous_count = 2540047
        manifest["matches_previous_local_count"] = (
            source_count == previous_count
        )

        if source_count != previous_count:
            raise ValueError(
                f"Previous local audit counted {previous_count:,} rows, "
                f"but this run loaded {source_count:,}. "
                "Check the source files before proceeding."
            )

        manifest["records"] = source_count
        manifest["columns"] = len(source.columns)
        manifest["class_counts"] = [
            row.asDict()
            for row in source.groupBy("label")
            .count()
            .orderBy("label")
            .collect()
        ]

        (results_dir / "full_schema.json").write_text(
            source.schema.json(),
            encoding="utf-8",
        )

        storage_rows.append({
            "format": "csv",
            "records": source_count,
            "data_bytes": input_bytes,
            "metadata_bytes": 0,
            "auxiliary_bytes": 0,
            "total_bytes": input_bytes,
        })

        for fmt in ("parquet", "delta"):
            target = storage_dir / fmt
            target_uri = target.as_uri()

            print(f"\nWriting {fmt.upper()}...", flush=True)

            start = perf_counter()

            (
                source.repartition(4)
                .write.format(fmt)
                .option("compression", "snappy")
                .mode("errorifexists")
                .save(target_uri)
            )

            write_seconds = perf_counter() - start

            restored = spark.read.format(fmt).load(target_uri)

            if column_types(source) != column_types(restored):
                raise RuntimeError(
                    f"{fmt}: column names or data types changed."
                )

            restored = restored.select(*source.columns)
            restored_count = restored.count()

            print("Checking missing rows...", flush=True)
            missing_rows = source.exceptAll(restored).count()

            print("Checking extra rows...", flush=True)
            extra_rows = restored.exceptAll(source).count()

            passed = (
                restored_count == source_count
                and missing_rows == 0
                and extra_rows == 0
            )

            sizes = measure_storage(target)

            manifest["formats"].append({
                "format": fmt,
                "path": str(target),
                "records": restored_count,
                "missing_rows": missing_rows,
                "extra_rows": extra_rows,
                "verified": passed,
                "write_seconds_single_observation": write_seconds,
                **sizes,
            })

            if not passed:
                raise RuntimeError(f"{fmt}: conversion verification failed.")

            storage_rows.append({
                "format": fmt,
                "records": restored_count,
                **sizes,
            })

            print(
                f"PASS: full / {fmt} | "
                f"rows={restored_count:,} | "
                f"missing={missing_rows} | extra={extra_rows}",
                flush=True,
            )

            print(
                f"Directory file size: "
                f"{sizes['total_bytes'] / (1024 ** 2):.2f} MiB",
                flush=True,
            )

        with (results_dir / "storage_sizes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(storage_rows[0]),
            )
            writer.writeheader()
            writer.writerows(storage_rows)

        manifest["status"] = "passed"

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise

    finally:
        manifest_path = results_dir / "full_conversion_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print("\nManifest:", manifest_path, flush=True)

        if spark is not None:
            try:
                if source is not None:
                    source.unpersist()
            finally:
                spark.stop()

    print("\nFULL DATASET CONVERSION PASSED")
    print("Storage folder:", storage_dir)
    print("Results folder:", results_dir)


if __name__ == "__main__":
    main()