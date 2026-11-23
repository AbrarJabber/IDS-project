
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

spark = (
    SparkSession.builder
    .appName("UNSW-NB15-Logistic-Regression")
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

    # Text columns that must be encoded
    categorical_columns = [
        "proto", "service", "state"
    ]

    # Exclude both target columns
    numeric_columns = [
        c for c in train_df.columns
        if c not in categorical_columns
        and c not in ["label", "attack_cat"]
    ]

    stages = []

    # Convert categorical strings into indices
    for column_name in categorical_columns:
        stages.append(
            StringIndexer(
                inputCol=column_name,
                outputCol=column_name + "_index",
                handleInvalid="keep"
            )
        )

    # Convert category indices into one-hot vectors
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

    # Combine all features
    stages.append(
        VectorAssembler(
            inputCols=numeric_columns + [
                c + "_encoded" for c in categorical_columns
            ],
            outputCol="unscaled_features",
            handleInvalid="error"
        )
    )

    # Scale features for Logistic Regression
    stages.append(
        StandardScaler(
            inputCol="unscaled_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )
    )

    # Logistic Regression classifier
    stages.append(
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            regParam=0.01
        )
    )

    pipeline = Pipeline(stages=stages)

    print("\nTraining Logistic Regression...")
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

    print("\nLOGISTIC REGRESSION COMPLETED")

finally:
    spark.stop()
