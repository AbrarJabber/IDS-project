
import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print("Python:", sys.executable, flush=True)

spark = (
    SparkSession.builder
    .appName("IDS-Test")
    .master("local[1]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("TEST 1: Spark without Python workers", flush=True)
spark.range(4).show()

print("TEST 2: Spark with Python workers", flush=True)
df = spark.createDataFrame(
    [("Normal", 100), ("Attack", 250)],
    ["Traffic_Type", "Packets"]
)
df.show()

print("SPARK TEST SUCCESSFUL", flush=True)

spark.stop()

