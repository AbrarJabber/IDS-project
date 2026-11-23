
from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    count,
    sum as spark_sum
)

from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

SELECTED_THRESHOLD = 0.60


spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Final-Random-Forest")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def prepare_data(df):
    return (
        df.drop("id")
        .withColumn(
            "service",
            when(
                col("service") == "-",
                "unknown"
            ).otherwise(col("service"))
        )
    )


def calculate_metrics(predictions, threshold):

    evaluated = predictions.withColumn(
        "adjusted_prediction",
        when(
            col("attack_probability") >= threshold,
            1
        ).otherwise(0)
    )

    counts = (
        evaluated.groupBy(
            "label",
            "adjusted_prediction"
        )
        .count()
        .collect()
    )

    results = {
        (
            int(row["label"]),
            int(row["adjusted_prediction"])
        ): row["count"]
        for row in counts
    }

    tn = results.get((0, 0), 0)
    fp = results.get((0, 1), 0)
    fn = results.get((1, 0), 0)
    tp = results.get((1, 1), 0)

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def print_metrics(result):

    print(f"\nThreshold: {result['threshold']:.2f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Attack Precision: {result['precision']:.4f}")
    print(f"Attack Recall: {result['recall']:.4f}")
    print(f"Attack F1-score: {result['f1']:.4f}")
    print(f"False Positive Rate: {result['fpr']:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives: {result['tn']}")
    print(f"False Positives: {result['fp']}")
    print(f"False Negatives: {result['fn']}")
    print(f"True Positives: {result['tp']}")


try:

    print("\nLoading datasets...")

    train_df = prepare_data(
        spark.read.csv(
            str(DATA_DIR / "UNSW_NB15_training-set.csv"),
            header=True,
            inferSchema=True
        )
    )

    test_df = prepare_data(
        spark.read.csv(
            str(DATA_DIR / "UNSW_NB15_testing-set.csv"),
            header=True,
            inferSchema=True
        )
    )

    print("Training records:", train_df.count())
    print("Testing records:", test_df.count())

    categorical_columns = [
        "proto",
        "service",
        "state"
    ]

    numeric_columns = [
        c for c in train_df.columns
        if c not in categorical_columns
        and c not in ["label", "attack_cat"]
    ]

    stages = []

    for column_name in categorical_columns:

        stages.append(
            StringIndexer(
                inputCol=column_name,
                outputCol=column_name + "_index",
                handleInvalid="keep"
            )
        )

    stages.append(
        OneHotEncoder(
            inputCols=[
                c + "_index"
                for c in categorical_columns
            ],
            outputCols=[
                c + "_encoded"
                for c in categorical_columns
            ],
            handleInvalid="keep"
        )
    )

    stages.append(
        VectorAssembler(
            inputCols=numeric_columns + [
                c + "_encoded"
                for c in categorical_columns
            ],
            outputCol="features",
            handleInvalid="error"
        )
    )

    stages.append(
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
    )

    pipeline = Pipeline(stages=stages)

    print("\nTraining final Random Forest...")

    training_start = time.perf_counter()

    model = pipeline.fit(train_df)

    training_time = time.perf_counter() - training_start

    print(f"Training time: {training_time:.2f} seconds")

    print("\nGenerating predictions...")

    prediction_start = time.perf_counter()

    predictions = (
        model.transform(test_df)
        .withColumn(
            "attack_probability",
            vector_to_array(col("probability"))[1]
        )
        .select(
            "label",
            "attack_cat",
            "attack_probability"
        )
        .cache()
    )

    # Materialize the predictions to measure execution time
    predictions.count()

    prediction_time = time.perf_counter() - prediction_start

    print(f"Prediction time: {prediction_time:.2f} seconds")

    print("\n=== STANDARD THRESHOLD (0.50) ===")

    standard_result = calculate_metrics(
        predictions,
        0.50
    )

    print_metrics(standard_result)

    print("\n=== SELECTED THRESHOLD (0.60) ===")

    final_result = calculate_metrics(
        predictions,
        SELECTED_THRESHOLD
    )

    print_metrics(final_result)

    print("\n=== ROC AUC ===")

    # ROC AUC can be calculated from attack probability
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="attack_probability",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(predictions)

    print(f"ROC AUC: {auc:.4f}")

    print("\n=== ATTACK CATEGORY ANALYSIS ===")

    evaluated = predictions.withColumn(
        "adjusted_prediction",
        when(
            col("attack_probability") >= SELECTED_THRESHOLD,
            1
        ).otherwise(0)
    )

    category_results = (
        evaluated
        .filter(col("label") == 1)
        .groupBy("attack_cat")
        .agg(
            count("*").alias("total_attacks"),
            spark_sum(
                when(
                    col("adjusted_prediction") == 1,
                    1
                ).otherwise(0)
            ).alias("detected"),
            spark_sum(
                when(
                    col("adjusted_prediction") == 0,
                    1
                ).otherwise(0)
            ).alias("missed")
        )
        .withColumn(
            "detection_rate",
            col("detected") / col("total_attacks")
        )
        .orderBy("detection_rate")
    )

    category_results.show(
        20,
        truncate=False
    )

    print("\n=== EXPERIMENT SUMMARY ===")

    print(f"Training records: {train_df.count()}")
    print(f"Testing records: {test_df.count()}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Selected threshold: {SELECTED_THRESHOLD}")
    print(f"Final accuracy: {final_result['accuracy']:.4f}")
    print(f"Final F1-score: {final_result['f1']:.4f}")
    print(f"Final ROC AUC: {auc:.4f}")

    predictions.unpersist()

    print("\nFINAL RANDOM FOREST EXPERIMENT COMPLETED")

finally:

    spark.stop()
