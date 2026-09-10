from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def load_and_split_data(spark: SparkSession, data_path: str):
    """Loads ML-1M data and splits chronologically by strict calendar time duration (70% / 15% / 15%)."""
    df = spark.read.csv(data_path, sep="::", inferSchema=False) \
        .toDF("userId", "movieId", "rating", "timestamp") \
        .select(
            F.col("userId").cast("integer"),
            F.col("movieId").cast("integer"),
            F.col("rating").cast("float"),
            F.col("timestamp").cast("long")
        )

    # 1. Compute global start and end timestamps
    bounds = df.agg(
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts")
    ).collect()[0]

    min_ts = bounds["min_ts"]
    max_ts = bounds["max_ts"]
    total_duration = max_ts - min_ts

    # 2. Compute calendar cutoffs based on total duration span
    t_70 = min_ts + int(total_duration * 0.70)
    t_85 = min_ts + int(total_duration * 0.85)

    # 3. Filter chronologically by timestamp thresholds
    train_df = df.filter(F.col("timestamp") <= t_70)
    val_df   = df.filter((F.col("timestamp") > t_70) & (F.col("timestamp") <= t_85))
    test_df  = df.filter(F.col("timestamp") > t_85)

    return train_df.cache(), val_df.cache(), test_df.cache()