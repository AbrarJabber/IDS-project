
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Start Apache Spark
spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Exploration")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Dataset locations
train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

# Check that files exist
if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        "Dataset files not found. Check the data/raw folder."
    )

# Load datasets
train_df = spark.read.csv(
    str(train_path),
    header=True,
    inferSchema=True
)

test_df = spark.read.csv(
    str(test_path),
    header=True,
    inferSchema=True
)

# 1. Dataset dimensions
print("\n=== DATASET DIMENSIONS ===")
print("Training records:", train_df.count())
print("Testing records:", test_df.count())
print("Number of columns:", len(train_df.columns))

# 2. Preview the dataset
print("\n=== FIRST FIVE RECORDS ===")
train_df.show(5, truncate=False)

# 3. Column types
print("\n=== DATASET SCHEMA ===")
train_df.printSchema()

# 4. Normal vs attack traffic
print("\n=== NORMAL VS ATTACK TRAFFIC ===")
train_df.groupBy("label").count().orderBy("label").show()

# 5. Attack categories
print("\n=== ATTACK CATEGORY DISTRIBUTION ===")
train_df.groupBy("attack_cat").count().orderBy(
    col("count").desc()
).show(truncate=False)

# 6. Protocol distribution
print("\n=== NETWORK PROTOCOL DISTRIBUTION ===")
train_df.groupBy("proto").count().orderBy(
    col("count").desc()
).show(15, truncate=False)

# 7. Missing values
print("\n=== MISSING VALUES ===")
missing_counts = train_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in train_df.columns
])
missing_counts.show(vertical=True, truncate=False)

# Stop Spark
spark.stop()

print("\nDATA EXPLORATION COMPLETED")
