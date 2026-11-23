
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Random-Forest")
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

    categorical_columns = [
        "proto", "service", "state"
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
                c + "_index" for c in categorical_columns
            ],
            outputCols=[
                c + "_encoded" for c in categorical_columns
            ],
            handleInvalid="keep"
        )
    )

    stages.append(
        VectorAssembler(
            inputCols=numeric_columns + [
                c + "_encoded" for c in categorical_columns
            ],
            outputCol="features"
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
    model = pipeline.fit(train_df)

    print("\nMaking predictions...")
    predictions = model.transform(test_df)

    print("\n=== CONFUSION MATRIX ===")
    predictions.groupBy(
        "label", "prediction"
    ).count().orderBy(
        "label", "prediction"
    ).show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction"
    )

    for metric in [
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "f1"
    ]:
        score = evaluator.evaluate(
            predictions,
            {evaluator.metricName: metric}
        )
        print(f"{metric}: {score:.4f}")

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(predictions)
    print(f"ROC AUC: {auc:.4f}")

    print("\nRANDOM FOREST COMPLETED")

finally:
    spark.stop()
