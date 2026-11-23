"""
Compare CSV, Parquet, and Delta using identical Spark queries.

Design:
- One Spark session.
- Explicit CSV schema from the verified conversion.
- One excluded warm-up per query and format.
- Six measured repetitions per query and format.
- Balanced format execution order.
- No explicit Spark DataFrame caching.
- Check every query result against the CSV reference.

Timings include reader construction, query construction, and collect().
Spark startup, verification, and output-file writing are excluded.

Operating-system and internal metadata caches are NOT cleared.
These are warmed-session timings, not cold-disk measurements.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import statistics
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType


FORMATS = ("csv", "parquet", "delta")
QUERIES = ("protocol_label", "tcp_label")


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
        part.lower() for part in current_path.split(os.pathsep)
    ]:
        os.environ["PATH"] = bin_path + os.pathsep + current_path


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_result(rows):
    """Compare small aggregate results without depending on row order."""
    return sorted(
        json.dumps(row.asDict(), sort_keys=True)
        for row in rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--conversion-dir",
        type=Path,
        required=True,
        help="Results directory produced by script 20.",
    )

    args = parser.parse_args()
    conversion_dir = args.conversion_dir.resolve()

    conversion = json.loads(
        (conversion_dir / "full_conversion_manifest.json")
        .read_text(encoding="utf-8")
    )

    if conversion.get("status") != "passed":
        raise ValueError("The full conversion must have passed first.")

    schema = StructType.fromJson(
        json.loads(
            (conversion_dir / "full_schema.json")
            .read_text(encoding="utf-8")
        )
    )

    csv_paths = [Path(item["path"]) for item in conversion["inputs"]]

    format_paths = {
        item["format"]: Path(item["path"])
        for item in conversion["formats"]
        if item.get("verified") is True
    }

    if not {"parquet", "delta"}.issubset(format_paths):
        raise ValueError("Verified Parquet and Delta paths are required.")

    for item in conversion["inputs"]:
        path = Path(item["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"CSV size changed since conversion: {path}")

    for path in format_paths.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    configure_windows()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    root = Path(__file__).resolve().parents[2]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = root / "big_data/results/storage_benchmark" / run_id
    output.mkdir(parents=True, exist_ok=False)

    # All six permutations: each format occupies each position twice.
    orders = list(itertools.permutations(FORMATS))
    random.Random(42).shuffle(orders)

    manifest = {
        "status": "running",
        "python": platform.python_version(),
        "pyspark": version("pyspark"),
        "delta_spark": version("delta-spark"),
        "platform": platform.platform(),
        "conversion_directory": str(conversion_dir),
        "input_records": conversion["records"],
        "input_schema": schema.jsonValue(),
        "csv_inputs": [str(path) for path in csv_paths],
        "format_paths": {
            key: str(value) for key, value in format_paths.items()
        },
        "delta_version": 0,
        "seed": 42,
        "warmups_per_query_format": 1,
        "measured_repetitions_per_query_format": 6,
        "format_orders": orders,
        "timing_scope": (
            "Reader construction through completed aggregate collect; "
            "Spark startup, verification, and result-file writing excluded."
        ),
        "cache_policy": (
            "No explicit DataFrame caching. OS and internal metadata "
            "caches are not cleared. Warmed single-session experiment."
        ),
        "notes": [
            "Full dataset is stored as strings in all formats.",
            "Identical numeric casts are applied within every query.",
            "Natural input partitioning can differ between formats.",
            "Input-size checks do not constitute file-content hashes.",
        ],
    }

    measurements = []
    reference = {}
    spark = None

    builder = (
        SparkSession.builder
        .appName("IDS_Storage_Format_Benchmark")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.ansi.enabled", "false")
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

        manifest["spark_settings"] = {
            key: spark.conf.get(key)
            for key in (
                "spark.sql.shuffle.partitions",
                "spark.sql.adaptive.enabled",
                "spark.sql.ansi.enabled",
                "spark.sql.files.maxPartitionBytes",
                "spark.sql.parquet.filterPushdown",
            )
        }
        manifest["spark_master"] = spark.sparkContext.master

        def read_data(fmt):
            if fmt == "csv":
                return spark.read.csv(
                    [path.as_uri() for path in csv_paths],
                    schema=schema,
                    header=False,
                    mode="FAILFAST",
                )

            reader = spark.read.format(fmt)

            if fmt == "delta":
                # Pin the verified initial snapshot.
                reader = reader.option("versionAsOf", 0)

            return reader.load(format_paths[fmt].as_uri())

        def execute_query(fmt, query_name):
            spark.catalog.clearCache()

            start = perf_counter()

            df = read_data(fmt)

            if query_name == "tcp_label":
                df = df.filter(F.col("proto") == "tcp")
                grouping = ["label"]
            else:
                grouping = ["proto", "label"]

            # Same projection and casts for all storage formats.
            df = df.select(
                "proto",
                "label",
                F.col("sbytes").cast("long").alias("source_bytes"),
                F.col("dbytes").cast("long").alias("destination_bytes"),
            )

            result = (
                df.groupBy(*grouping)
                .agg(
                    F.count("*").alias("records"),
                    F.sum("source_bytes").alias("total_source_bytes"),
                    F.sum("destination_bytes").alias(
                        "total_destination_bytes"
                    ),
                    F.count("source_bytes").alias("valid_source_bytes"),
                    F.count("destination_bytes").alias(
                        "valid_destination_bytes"
                    ),
                )
                .collect()
            )

            elapsed = perf_counter() - start
            return result, elapsed

        def verify(query_name, rows):
            signature = canonical_result(rows)

            if signature != reference[query_name]:
                raise RuntimeError(
                    f"Result mismatch for query: {query_name}"
                )

            return hashlib.sha256(
                "\n".join(signature).encode("utf-8")
            ).hexdigest()

        print("\n=== EXCLUDED WARM-UP RUNS ===", flush=True)

        # CSV establishes the reference for each query.
        for query_name in QUERIES:
            for fmt in FORMATS:
                rows, _ = execute_query(fmt, query_name)

                if fmt == "csv":
                    if not rows:
                        raise RuntimeError(
                            f"Query returned no groups: {query_name}"
                        )

                    if query_name == "protocol_label":
                        counted = sum(row["records"] for row in rows)

                        if counted != conversion["records"]:
                            raise RuntimeError(
                                "CSV query count differs from conversion."
                            )

                    reference[query_name] = canonical_result(rows)

                    (output / f"reference_{query_name}.json").write_text(
                        json.dumps(
                            [row.asDict() for row in rows],
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                verify(query_name, rows)
                print(f"Warm-up passed: {query_name} / {fmt}", flush=True)

        print("\n=== MEASURED RUNS ===", flush=True)

        for repetition, order in enumerate(orders, start=1):
            print(
                f"\nRound {repetition}/6: {' -> '.join(order)}",
                flush=True,
            )

            # Alternate query order between rounds.
            query_order = (
                QUERIES if repetition % 2 else tuple(reversed(QUERIES))
            )

            for position, fmt in enumerate(order, start=1):
                for query_name in query_order:
                    rows, elapsed = execute_query(fmt, query_name)
                    result_hash = verify(query_name, rows)

                    measurements.append({
                        "query": query_name,
                        "format": fmt,
                        "repetition": repetition,
                        "format_position": position,
                        "seconds": elapsed,
                        "result_groups": len(rows),
                        "result_sha256": result_hash,
                        "verified": True,
                    })

                    print(
                        f"{query_name} | {fmt} | "
                        f"{elapsed:.4f} seconds | verified",
                        flush=True,
                    )

            # Preserve completed rounds as the experiment progresses.
            save_csv(output / "benchmark_runs.csv", measurements)

        summaries = []

        for query_name in QUERIES:
            csv_mean = statistics.mean(
                row["seconds"]
                for row in measurements
                if row["query"] == query_name and row["format"] == "csv"
            )

            for fmt in FORMATS:
                times = [
                    row["seconds"]
                    for row in measurements
                    if row["query"] == query_name and row["format"] == fmt
                ]

                mean_seconds = statistics.mean(times)

                summaries.append({
                    "query": query_name,
                    "format": fmt,
                    "repetitions": len(times),
                    "mean_seconds": mean_seconds,
                    "stddev_seconds": statistics.stdev(times),
                    "median_seconds": statistics.median(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "speedup_vs_csv": csv_mean / mean_seconds,
                })

        save_csv(output / "benchmark_summary.csv", summaries)

        print("\n=== SUMMARY ===", flush=True)

        for row in summaries:
            print(
                f"{row['query']} | {row['format']} | "
                f"mean={row['mean_seconds']:.4f}s | "
                f"SD={row['stddev_seconds']:.4f}s | "
                f"speedup={row['speedup_vs_csv']:.2f}x"
            )

        manifest["status"] = "passed"
        manifest["measured_runs"] = len(measurements)

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise

    finally:
        save_csv(output / "benchmark_runs.csv", measurements)

        (output / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        if spark is not None:
            spark.stop()

    print("\nSTORAGE BENCHMARK PASSED")
    print("Measured runs:", len(measurements))
    print("Results:", output)


if __name__ == "__main__":
    main()