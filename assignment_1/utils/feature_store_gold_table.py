import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

from pyspark.ml.linalg import SparseVector
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import col, row_number, to_date, count, percentile_approx, desc, udf, collect_set, array_contains, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType, ArrayType, NumericType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import regexp_extract, regexp_replace, when, trim, initcap, round, lower, split, explode, array_distinct, array_sort, concat_ws, expr


def feature_store_gold_table(snapshot_date_str, silver_feature_store_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_feature_store_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_feature_store_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # OHE
    # Occupation
    # Index and encode
    indexer = StringIndexer(inputCol="Occupation", outputCol="Occupation_Indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="Occupation_Indexed", outputCol="Occupation_OHE")  # dropLast=True by default

    pipeline = Pipeline(stages=[indexer, encoder])
    model = pipeline.fit(df)
    df_encoded = model.transform(df)

    # Get occupation labels
    occupation_labels = model.stages[0].labels  # StringIndexer labels

    #  Convert OHE vector to array
    def vector_to_array(v):
        return v.toArray().tolist()

    vec_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
    df_encoded = df_encoded.withColumn("Occupation_OHE_Array", vec_to_array_udf(col("Occupation_OHE")))

    # Create individual binary columns, but DROP the first category (index 0)
    for i, label in enumerate(occupation_labels[1:], start=1):  # Skip first label
        col_name = f"Occupation_{label.replace(' ', '_')}"
        df_encoded = df_encoded.withColumn(col_name, col("Occupation_OHE_Array")[i - 1])  # Index i-1 due to dropped one

    # Clean up intermediate columns
    df_encoded = df_encoded.drop("Occupation_Indexed", "Occupation_OHE", "Occupation_OHE_Array", "Occupation")

    # Credit Mix
    # Index and encode
    indexer = StringIndexer(inputCol="Credit_Mix", outputCol="Credit_Mix_Indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="Credit_Mix_Indexed", outputCol="Credit_Mix_OHE")  # dropLast=True by default

    pipeline = Pipeline(stages=[indexer, encoder])
    model = pipeline.fit(df_encoded)
    df_encoded = model.transform(df_encoded)

    # Get credit mix labels
    Credit_Mix_labels = model.stages[0].labels  # StringIndexer labels

    #  Convert OHE vector to array
    def vector_to_array(v):
        return v.toArray().tolist()

    vec_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
    df_encoded = df_encoded.withColumn("Credit_Mix_OHE_Array", vec_to_array_udf(col("Credit_Mix_OHE")))

    # Create individual binary columns, but DROP the first category (index 0)
    for i, label in enumerate(Credit_Mix_labels[1:], start=1):  # Skip first label
        col_name = f"Credit_Mix_{label.replace(' ', '_')}"
        df_encoded = df_encoded.withColumn(col_name, col("Credit_Mix_OHE_Array")[i - 1])  # Index i-1 due to dropped one

    # Clean up intermediate columns
    df_encoded = df_encoded.drop("Credit_Mix_Indexed", "Credit_Mix_OHE", "Credit_Mix_OHE_Array", "Credit_Mix")

    # Payment Behaviour
    # Index and encode
    indexer = StringIndexer(inputCol="Payment_Behaviour", outputCol="Payment_Behaviour_Indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="Payment_Behaviour_Indexed", outputCol="Payment_Behaviour_OHE")  # dropLast=True by default

    pipeline = Pipeline(stages=[indexer, encoder])
    model = pipeline.fit(df_encoded)
    df_encoded = model.transform(df_encoded)

    # Get occupation labels
    Payment_Behaviour_labels = model.stages[0].labels  # StringIndexer labels

    #  Convert OHE vector to array
    def vector_to_array(v):
        return v.toArray().tolist()

    vec_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
    df_encoded = df_encoded.withColumn("Payment_Behaviour_OHE_Array", vec_to_array_udf(col("Payment_Behaviour_OHE")))

    # Create individual binary columns, but DROP the first category (index 0)
    for i, label in enumerate(Payment_Behaviour_labels[1:], start=1):  # Skip first label
        col_name = f"Payment_Behaviour_{label.replace(' ', '_')}"
        df_encoded = df_encoded.withColumn(col_name, col("Payment_Behaviour_OHE_Array")[i - 1])  # Index i-1 due to dropped one

    # Clean up intermediate columns
    df_encoded = df_encoded.drop("Payment_Behaviour_Indexed", "Payment_Behaviour_OHE", "Payment_Behaviour_OHE_Array", "Payment_Behaviour")

    # # Loan

    # df_encoded = df_encoded.withColumn(
    # "Loan",
    # when(col("Loan") == "Unknown", "Not Specified")
    # .otherwise(col("Loan"))
    # )

    # # 4. Get all unique loan types (including "Unknown")
    # loan_types = df_encoded.select("Loan").distinct().rdd.flatMap(lambda x: x).filter(lambda x: x.strip() != "") .collect()

    # # 5. One-hot encode: add a column for each unique loan type
    # for loan in loan_types:
    #     clean_col_name = loan.replace(" ", "_")
    #     df_encoded = df_encoded.withColumn(
    #         clean_col_name,
    #         (array_contains(col("Loan_List"), loan)).cast("int")
    #     )

    # # 6. Drop intermediate columns
    # df_encoded = df_encoded.drop("Loan", "Loan_List")
    # df_encoded = df_encoded.withColumnRenamed('Not_Specified', 'Not_Specified_Loan')
    # df_encoded = df_encoded.drop('Not_Specified_Loan')

    # Payment of Min Amount
    # Index and encode
    indexer = StringIndexer(inputCol="Payment_of_Min_Amount", outputCol="Payment_of_Min_Amount_Indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="Payment_of_Min_Amount_Indexed", outputCol="Payment_of_Min_Amount_OHE")  # dropLast=True by default

    pipeline = Pipeline(stages=[indexer, encoder])
    model = pipeline.fit(df_encoded)
    df_encoded = model.transform(df_encoded)

    # Get occupation labels
    Payment_of_Min_Amount_labels = model.stages[0].labels  # StringIndexer labels

    #  Convert OHE vector to array
    def vector_to_array(v):
        return v.toArray().tolist()

    vec_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
    df_encoded = df_encoded.withColumn("Payment_of_Min_Amount_OHE_Array", vec_to_array_udf(col("Payment_of_Min_Amount_OHE")))

    # Create individual binary columns, but DROP the first category (index 0)
    for i, label in enumerate(Payment_of_Min_Amount_labels[1:], start=1):  # Skip first label
        col_name = f"Payment_of_Min_Amount_{label.replace(' ', '_')}"
        df_encoded = df_encoded.withColumn(col_name, col("Payment_of_Min_Amount_OHE_Array")[i - 1])  # Index i-1 due to dropped one

    # Clean up intermediate columns
    df_encoded = df_encoded.drop("Payment_of_Min_Amount_Indexed", "Payment_of_Min_Amount_OHE", "Payment_of_Min_Amount_OHE_Array", "Payment_of_Min_Amount")

    # Type of Loan
    # Step 3: Get all unique loan types
    all_loans = df_encoded.select(F.explode("loan_array").alias("loan_type")) \
                .distinct() \
                .collect()

    loan_types = [row.loan_type for row in all_loans]

    # Step 4: Create one-hot encoded columns
    for loan in loan_types:
        col_name = loan.replace(" ", "_").replace("-", "_")
        df_encoded = df_encoded.withColumn(
            col_name,
            F.array_contains("loan_array", loan).cast("int")
        )

    # Step 5: Clean up intermediate columns
    df_encoded = df_encoded.drop("cleaned_loans", "loan_array","not_specified", "Type_of_Loan")



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

    #Int_across_loans
    df = df.withColumn(
        "Int_across_loans",
        (col("Interest_Rate") / col("Num_of_Loan"))
    )

    #Delinquency_Score
    df = df.withColumn(
        "Delinquency_Score",
        (col("Num_of_Delayed_Payment") / (col("Num_Credit_Card")+col("Num_of_Loan"))
    ))

    # Round to 2dp
    df = df.withColumn("Income_to_Debt_Ratio", round(col("Income_to_Debt_Ratio"), 2)) \
        .withColumn("EMI_to_Income_Ratio", round(col("EMI_to_Income_Ratio"), 2)) \
        .withColumn("Monthly_Saving", round(col("Monthly_Saving"), 2)) \
        .withColumn("Int_across_loans", round(col("Int_across_loans"), 2)) \
        .withColumn("Delinquency_Score", round(col("Delinquency_Score"), 2))

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df