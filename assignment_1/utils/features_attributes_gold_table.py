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

from pyspark.sql.functions import col, row_number, to_date, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, ArrayType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def features_attributes_gold_table(snapshot_date_str, silver_features_attributes_directory, gold_features_attributes_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_features_attributes_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Impute occupation with most freq value and age with median val
    # Occupation
    # Find the most frequent (mode) value
    mode_value = (
        df.filter(col("Occupation").isNotNull())
        .groupBy("Occupation")
        .count()
        .orderBy(col("count").desc())
        .first()[0]
    )

    #  Fill nulls in the column with the mode value
    df_cleaned = df.fillna({ "Occupation": mode_value })

    # Age
    # Filter out nulls and select the column as a list of values
    non_null_values = df_cleaned.select("Age").where(col("Age").isNotNull())

    # Compute the approximate median using approxQuantile
    # [0.5] is for the median (50th percentile), and 0.01 is the relative error
    median_value = non_null_values.approxQuantile("Age", [0.5], 0.01)[0]

    # Fill nulls with the median
    df_cleaned = df_cleaned.fillna({ "Age": median_value })


    # Index and encode
    indexer = StringIndexer(inputCol="Occupation", outputCol="Occupation_Indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="Occupation_Indexed", outputCol="Occupation_OHE")  # dropLast=True by default

    pipeline = Pipeline(stages=[indexer, encoder])
    model = pipeline.fit(df_cleaned)
    df_encoded = model.transform(df_cleaned)

    # Get occupation labels
    occupation_labels = model.stages[0].labels  # StringIndexer labels

    #  Convert OHE vector to array
    def vector_to_array(v):
        return v.toArray().tolist()

    vec_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
    df_encoded = df_encoded.withColumn("Occupation_OHE_Array", vec_to_array_udf(col("Occupation_OHE")))

    # Create individual binary columns, but DROP the first category (index 0)
    for i, label in enumerate(occupation_labels[1:], start=1):  # Skip first label
        col_name = f"is_{label.replace(' ', '_')}"
        df_encoded = df_encoded.withColumn(col_name, col("Occupation_OHE_Array")[i - 1])  # Index i-1 due to dropped one

    # Clean up intermediate columns
    df_encoded = df_encoded.drop("Occupation_Indexed", "Occupation_OHE", "Occupation_OHE_Array", "Occupation")

    # Get only latest snapshot for ML
    # Ensure snapshot_date is in date format
    df_cleaned = df_encoded.withColumn("snapshot_date", to_date("snapshot_date", "yyyy-MM-dd"))

    # Define a window partitioned by Customer_ID, ordered by snapshot_date descending
    window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())

    # Add row_number to rank records
    ranked_df = df_cleaned.withColumn("rn", row_number().over(window_spec))

    # Filter only latest records (row_number == 1)
    df = ranked_df.filter(col("rn") == 1).drop("rn")


    # save gold table - IRL connect to database to write
    partition_name = "gold_features_attributes_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_features_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df