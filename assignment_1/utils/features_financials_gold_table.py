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

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

from pyspark.sql.functions import col, row_number, to_date, count, percentile_approx, desc, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType, ArrayType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.window import Window
from pyspark.sql.functions import regexp_extract, regexp_replace, when, trim, initcap, round, lower, split, explode, array_distinct, array_sort, concat_ws, expr

def features_financials_gold_table(snapshot_date_str, silver_features_financials_directory, gold_features_financials_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_features_financials_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_financials_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Imputation

    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": DoubleType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": DoubleType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Credit_History_Age": DoubleType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": DoubleType(),
        "Amount_invested_monthly": DoubleType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DoubleType(),      
        "snapshot_date": DateType(),
    }

    # List of excluded columns
    excluded_cols = ["Customer_ID", "snapshot_date"]

    # Identify column types (exclude the excluded columns)
    string_cols = [c for c, t in column_type_map.items() if isinstance(t, StringType) and c not in excluded_cols]
    numeric_cols = [c for c, t in column_type_map.items() if isinstance(t, (IntegerType, DoubleType)) and c not in excluded_cols]

    # Impute empty or space-only strings with the most frequent value
    df_imputed = df

    for col_name in string_cols:
        # Find the most frequent (mode) value in the column
        mode_row = df_imputed.groupBy(col_name).count().orderBy(F.desc("count")).first()
        mode_val = mode_row[0] if mode_row and mode_row[0] is not None else None

        # If a mode is found, replace empty or space-only strings with the mode value
        if mode_val:
            df_imputed = df_imputed.withColumn(
                col_name,
                when((col(col_name).isNull()) | (trim(col(col_name)) == ""), mode_val).otherwise(col(col_name))
            )
        else:
            # If no mode found (which shouldn't happen), replace with "Unknown"
            df_imputed = df_imputed.withColumn(
                col_name,
                when((col(col_name).isNull()) | (trim(col(col_name)) == ""), "Unknown").otherwise(col(col_name))
            )

    # Impute numeric columns with the median
    for col_name in numeric_cols:
        median = df_imputed.approxQuantile(col_name, [0.5], 0.01)[0]
        df_imputed = df_imputed.fillna({col_name: median})

    #One hot encoding
    categorical_cols = ["Type_of_Loan", "Credit_Mix", "Payment_Behaviour"]

    # Index and one-hot encode (with dropLast=True to mimic drop_first=True)
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=c+"_index", outputCol=c+"_ohe", dropLast=True) for c in categorical_cols]

    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df_imputed)
    df_encoded = model.transform(df_imputed)

    # Convert vector to array
    vector_to_array_udf = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

    # Split each one-hot vector into binary columns
    for c in categorical_cols:
        arr_col = c + "_arr"
        df_encoded = df_encoded.withColumn(arr_col, vector_to_array_udf(col(c + "_ohe")))

        # Get array length to know number of binary columns
        num_categories = df_encoded.select(arr_col).head()[arr_col].__len__()

        # Create binary columns (0/1)
        for i in range(num_categories):
            df_encoded = df_encoded.withColumn(f"{c}_{i}", col(arr_col)[i].cast("int"))

        # Drop original column + intermediates
        df_encoded = df_encoded.drop(c, c + "_index", c + "_ohe", arr_col)

    # Feature Engineering
    # recent delay is > 10 days, else 0
    df = df_encoded.withColumn(
        "Recent_Delays",
        when(col("Delay_from_due_date") > 10, 1).otherwise(0)
    )

    df = df.withColumn(
        "Income_to_Debt_Ratio",
        (col("Annual_Income") / col("Outstanding_Debt"))
    )

    df = df.withColumn(
        "EMI_to_Income_Ratio",
        (col("Total_EMI_per_month") / (col("Annual_Income") / 12))
    )

    df = df.withColumn(
        "Monthly_Saving",
        col("Monthly_Inhand_Salary") - col("Total_EMI_per_month") - col("Amount_invested_monthly")
    )

    # Add Num_Bank_Accounts + Num_Credit_Card and flag if above threshold
    df = df.withColumn(
        "Is_Multi_Borrower",
        when((col("Num_Bank_Accounts") + col("Num_Credit_Card")) > 6, 1).otherwise(0)
    )

    # Round to 2dp
    df = df.withColumn("Income_to_Debt_Ratio", round(col("Income_to_Debt_Ratio"), 2)) \
        .withColumn("EMI_to_Income_Ratio", round(col("EMI_to_Income_Ratio"), 2)) \
        .withColumn("Monthly_Saving", round(col("Monthly_Saving"), 2))


    # Get only latest snapshot for ML
    # Ensure snapshot_date is in date format
    df_cleaned = df.withColumn("snapshot_date", to_date("snapshot_date", "yyyy-MM-dd"))

    # Define a window partitioned by Customer_ID, ordered by snapshot_date descending
    window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())

    # Add row_number to rank records
    ranked_df = df_cleaned.withColumn("rn", row_number().over(window_spec))

    # Filter only latest records (row_number == 1)
    df = ranked_df.filter(col("rn") == 1).drop("rn")


    # save gold table - IRL connect to database to write
    partition_name = "gold_features_financials_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_features_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df