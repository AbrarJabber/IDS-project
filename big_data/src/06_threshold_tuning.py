
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
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


spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Threshold-Tuning")
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

    total = tp + tn + fp + fn

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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": false_positive_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def print_results(result):

    print(f"Threshold: {result['threshold']:.2f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-score: {result['f1']:.4f}")
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

    print("\nSplitting training and validation datasets...")

    model_train, validation = train_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("Model training records:", model_train.count())
    print("Validation records:", validation.count())
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

    print("\nTraining Random Forest...")

    model = pipeline.fit(model_train)

    print("\nGenerating validation predictions...")

    validation_predictions = (
        model.transform(validation)
        .withColumn(
            "attack_probability",
            vector_to_array(col("probability"))[1]
        )
        .select(
            "label",
            "attack_probability"
        )
        .cache()
    )

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95
    ]

    validation_results = []

    print("\n=== VALIDATION RESULTS ===")

    print(
        "Threshold | Accuracy | Precision | "
        "Recall | F1 | FPR"
    )

    for threshold in thresholds:

        result = calculate_metrics(
            validation_predictions,
            threshold
        )

        validation_results.append(result)

        print(
            f"{threshold:.2f} | "
            f"{result['accuracy']:.4f} | "
            f"{result['precision']:.4f} | "
            f"{result['recall']:.4f} | "
            f"{result['f1']:.4f} | "
            f"{result['fpr']:.4f}"
        )

    best = max(
        validation_results,
        key=lambda result: result["f1"]
    )

    print("\n=== BEST VALIDATION THRESHOLD ===")

    print_results(best)

    validation_predictions.unpersist()

    print("\nGenerating test predictions...")

    test_predictions = (
        model.transform(test_df)
        .withColumn(
            "attack_probability",
            vector_to_array(col("probability"))[1]
        )
        .select(
            "label",
            "attack_probability"
        )
        .cache()
    )

    print("\n=== FINAL TEST RESULTS ===")

    final_result = calculate_metrics(
        test_predictions,
        best["threshold"]
    )

    print_results(final_result)

    print("\n=== TEST ROC AUC ===")

    # ROC AUC evaluates ranking quality across thresholds.
    # Use the model's raw predictions for this metric.
    auc_predictions = model.transform(test_df)

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(auc_predictions)

    print(f"ROC AUC: {auc:.4f}")

    test_predictions.unpersist()

    print("\nTHRESHOLD TUNING COMPLETED SUCCESSFULLY")

finally:

    spark.stop()
