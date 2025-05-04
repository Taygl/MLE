import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.window import Window

def feature_clickstream_gold_table(snapshot_date_str, silver_clickstream_directory, gold_clickstream_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Feature engineering
    # List of feature columns
    feature_cols = [f"fe_{i}" for i in range(1, 21)]

    # Keep only latest record for each customer
    # Latest Snapshot Features (per customer)
    latest_window = Window.partitionBy("Customer_ID").orderBy(F.col("snapshot_date").desc())
    
    latest_df = df.withColumn("row_num", F.row_number().over(latest_window)) \
                  .filter(F.col("row_num") == 1) \
                  .drop("row_num")

    # Aggregated Features (per customer across all time)
    agg_exprs = []
    for f_col in feature_cols:
        agg_exprs.extend([
            F.mean(f_col).alias(f"{f_col}_mean"),
            F.stddev(f_col).alias(f"{f_col}_stddev"),
            F.min(f_col).alias(f"{f_col}_min"),
            F.max(f_col).alias(f"{f_col}_max"),
        ])
    
    agg_df = df.groupBy("Customer_ID").agg(*agg_exprs)

    #Lag/Change Features (difference from previous snapshot)
    lag_window = Window.partitionBy("Customer_ID").orderBy("snapshot_date")
    
    df_with_lags = df
    for f_col in feature_cols:
        df_with_lags = df_with_lags.withColumn(f"{f_col}_lag", F.lag(f_col).over(lag_window))
        df_with_lags = df_with_lags.withColumn(f"{f_col}_change", F.col(f_col) - F.col(f"{f_col}_lag"))
    
    # Get the most recent change row per customer
    latest_changes = df_with_lags.withColumn("row_num", F.row_number().over(latest_window)) \
                                 .filter("row_num = 1") \
                                 .select("Customer_ID", *[f"{c}_change" for c in feature_cols])

    # Combine all features
    # Start with most recent snapshot (base)
    features_df = latest_df.select("Customer_ID", *feature_cols)
    
    # Join all blocks
    df = features_df \
        .join(agg_df, on="Customer_ID", how="left") \
        .join(latest_changes, on="Customer_ID", how="left")


    # save gold table - IRL connect to database to write
    partition_name = "gold_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df