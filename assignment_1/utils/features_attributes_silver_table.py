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
from pyspark.sql.functions import regexp_extract, regexp_replace, when, trim, initcap

def features_attributes_silver_table(snapshot_date_str, bronze_features_attributes_directory, silver_features_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_attributes_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # 1. Clean Age: Extract digits and cast to Integer
    df_cleaned = df.withColumn(
        "Age",
        regexp_extract(col("Age"), r"(\d+)", 1).cast(IntegerType())
    )

    # 2. Clean SSN: Match valid SSNs, else null
    valid_ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df_cleaned = df_cleaned.withColumn(
        "SSN",
        when(col("SSN").rlike(valid_ssn_pattern), col("SSN")).otherwise(None)
    )

    # 3. Clean Occupation: Remove placeholder, null if empty or invalid
    df_cleaned = df_cleaned.withColumn(
        "Occupation",
        when((trim(col("Occupation")) == "") | (col("Occupation") == "_______"), None)
        .otherwise(initcap(regexp_replace(col("Occupation"), "_", " ")))  
    )

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    #change to new dtype
    for column, new_type in column_type_map.items():
        df = df_cleaned.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attributes_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df