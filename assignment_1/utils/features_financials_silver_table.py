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

from pyspark.sql.functions import col, row_number, to_date, count, percentile_approx, desc, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType, ArrayType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.window import Window
from pyspark.sql.functions import regexp_extract, regexp_replace, when, trim, initcap, round, lower, split, explode, array_distinct, array_sort, concat_ws, expr

def features_financials_silver_table(snapshot_date_str, bronze_features_financials_directory, silver_features_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_financials_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean annual income
    # Remove non-numeric characters
    df_cleaned = df.withColumn(
        "Annual_Income",
        regexp_replace(col("Annual_Income"), r"[^\d.]", "")
        .cast("double")  # or "decimal(10,2)" for stricter typing
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn("Annual_Income", round(col("Annual_Income"), 2))

    # Monthly_Inhand_Salary
    # Clean non-numeric characters (keep digits and dot)
    df_cleaned = df_cleaned.withColumn(
        "Monthly_Inhand_Salary",
        regexp_replace(col("Monthly_Inhand_Salary"), r"[^\d.]", "").cast("double")
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn(
        "Monthly_Inhand_Salary",
        round(col("Monthly_Inhand_Salary"), 2)
    )

    # Clean Num_Bank_Accounts
    # Convert to integer and handle invalid values
    df_cleaned = df_cleaned.withColumn(
        "Num_Bank_Accounts",
        when(col("Num_Bank_Accounts").cast("int") < 0, None)  # Set negative values to null
        .otherwise(col("Num_Bank_Accounts").cast("int"))
    )

    # Clean Num_of_Loan
    # Remove non-digit characters and cast to integer
    df_cleaned = df_cleaned.withColumn(
        "Num_of_Loan",
        regexp_replace(col("Num_of_Loan"), r"[^\d\-]", "").cast("int")
    )

    # Replace invalid values (e.g. negative loans) with null
    df_cleaned = df_cleaned.withColumn(
        "Num_of_Loan",
        when(col("Num_of_Loan") < 0, None).otherwise(col("Num_of_Loan"))
    )

    # Type_of_Loan
    # Clean up type of loan and split them nicely
    # 1. Fill null or empty strings with "Not_Specified"
    df_filled = df_cleaned.fillna("Not Specified", subset=["Type_of_Loan"])

    # Replace empty strings in 'Type_of_Loan' with 'Not Specified'
    df_filled = df_filled.withColumn(
        "Type_of_Loan",
        when(trim(col("Type_of_Loan")) == "", "Not Specified").otherwise(col("Type_of_Loan"))
    )


    # 2. Clean: replace "and" with commas, split, and trim
    df_cleaned = df_filled.withColumn(
        "Cleaned_Loans",
        regexp_replace("Type_of_Loan", r"\s*and\s*", ",")
    )

    df_split = df_cleaned.withColumn(
        "Loan_List",
        split(regexp_replace(col("Cleaned_Loans"), r",\s*", ","), ",")
    )

    # 3. Explode to get distinct loan types
    df_exploded = df_split.withColumn("Loan", explode(col("Loan_List"))).withColumn("Loan", trim(col("Loan")))

    df_cleaned = df_exploded.drop("Type_of_Loan","Cleaned_Loans")

    # # Split on ',' and 'and', normalize separators
    # df_cleaned = df_cleaned.withColumn(
    #     "Type_of_Loan_Array",
    #     split(
    #         regexp_replace(col("Type_of_Loan"), r"\s*and\s*|\s*,\s*", ","), 
    #         ","
    #     )
    # )

    # # Trim entries, remove "Not Specified", deduplicate, sort
    # df_cleaned = df_cleaned.withColumn(
    #     "Type_of_Loan",
    #     concat_ws(
    #         ", ",
    #         array_sort(  # sort for consistency
    #             array_distinct(  # Remove duplicates
    #                 expr("filter(transform(Type_of_Loan_Array, x -> trim(x)), x -> x != 'Not Specified')")  # Remove 'Not Specified'
    #             )
    #         )
    #     )
    # )

    # # Drop the intermediate Type_of_Loan_Array column
    # df_cleaned = df_cleaned.drop("Type_of_Loan_Array")

    # Delay_from_due_date
    # Cast to integer and replace negative delays with null
    df_cleaned = df_cleaned.withColumn(
        "Delay_from_due_date",
        when(col("Delay_from_due_date").cast("int") < 0, None)
        .otherwise(col("Delay_from_due_date").cast("int"))
    )

    # Clean Num_of_Delayed_Payment
    # Remove non-digit characters and cast to integer
    df_cleaned = df_cleaned.withColumn(
        "Num_of_Delayed_Payment",
        regexp_replace(col("Num_of_Delayed_Payment"), r"[^\d\-]", "").cast("int")
    )

    # Replace invalid values (e.g. negative loans) with null
    df_cleaned = df_cleaned.withColumn(
        "Num_of_Delayed_Payment",
        when(col("Num_of_Delayed_Payment") < 0, None).otherwise(col("Num_of_Delayed_Payment"))
    )

    # Changed_Credit_Limit
    df_cleaned = df_cleaned.withColumn(
        "Changed_Credit_Limit",
        regexp_replace(col("Changed_Credit_Limit"), r"[^\d\.-]", "").cast("double")
    )

    df_cleaned = df_cleaned.withColumn(
        "Changed_Credit_Limit",
        round(col("Changed_Credit_Limit"), 2)
    )

    # Num_Credit_Inquiries
    # Cast to integer and replace negative delays with null
    df_cleaned = df_cleaned.withColumn(
        "Num_Credit_Inquiries",
        when(col("Num_Credit_Inquiries").cast("int") < 0, None)
        .otherwise(col("Num_Credit_Inquiries").cast("int"))
    )

    # Credit_Mix
    df_cleaned = df_cleaned.withColumn(
        "Credit_Mix",
        when(trim(col("Credit_Mix")) == "_", None)  # Replace "_" with null
        .otherwise(initcap(trim(col("Credit_Mix"))))  # Standardize casing
    )

    # Outstanding_Debt
    # Clean non-numeric characters (keep digits and dot)
    df_cleaned = df_cleaned.withColumn(
        "Outstanding_Debt",
        regexp_replace(col("Outstanding_Debt"), r"[^\d.]", "").cast("double")
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn(
        "Outstanding_Debt",
        round(col("Outstanding_Debt"), 2)
    )

    # Credit_Utilization_Ratio
    # Clean non-numeric characters (keep digits and dot)
    df_cleaned = df_cleaned.withColumn(
        "Credit_Utilization_Ratio",
        regexp_replace(col("Credit_Utilization_Ratio"), r"[^\d.]", "").cast("double")
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn(
        "Credit_Utilization_Ratio",
        round(col("Credit_Utilization_Ratio"), 2)
    )

    # Credit_History_Age
    # Extract years and months using regex
    df_cleaned = df_cleaned.withColumn("Years", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast("int")) \
                .withColumn("Months", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast("int"))

    # Convert to float years: years + (months / 12), rounded to 2 decimals
    df_cleaned = df_cleaned.withColumn(
        "Credit_History_Age",
        round(col("Years") + (col("Months") / 12), 2)
    )

    # Drop intermediate columns if desired
    df_cleaned = df_cleaned.drop("Years", "Months")

    # Payment_of_Min_Amount
    # Clean and standardize
    df_cleaned = df_cleaned.withColumn(
        "Payment_of_Min_Amount",
        when(trim(lower(col("Payment_of_Min_Amount"))) == "yes", "Yes")
        .when(trim(lower(col("Payment_of_Min_Amount"))) == "no", "No")
        .otherwise(None)  # Replace NM or any unexpected values with null
    )

    # Total_EMI_per_month
    # Clean non-numeric characters (keep digits and dot)
    df_cleaned = df_cleaned.withColumn(
        "Total_EMI_per_month",
        regexp_replace(col("Total_EMI_per_month"), r"[^\d.]", "").cast("double")
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn(
        "Total_EMI_per_month",
        round(col("Total_EMI_per_month"), 2)
    )

    # Amount_invested_monthly
    # Clean non-numeric characters (keep digits and dot)
    df_cleaned = df_cleaned.withColumn(
        "Amount_invested_monthly",
        regexp_replace(col("Amount_invested_monthly"), r"[^\d.]", "").cast("double")
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn(
        "Amount_invested_monthly",
        round(col("Amount_invested_monthly"), 2)
    )

    # Payment_Behaviour
    # Define valid patterns 
    valid_pattern = r"^(High|Low)_spent_(Small|Medium)_value_payments$"

    # Clean Payment_Behaviour
    df_cleaned = df_cleaned.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour").rlike(valid_pattern), col("Payment_Behaviour"))
        .otherwise(None)  # Set invalid values to null
    )

    # Monthly_Balance
    # Clean non-numeric characters (keep digits and dot)
    df_cleaned = df_cleaned.withColumn(
        "Monthly_Balance",
        regexp_replace(col("Monthly_Balance"), r"[^\d.]", "").cast("double")
    )

    # Round to 2 decimal places
    df_cleaned = df_cleaned.withColumn(
        "Monthly_Balance",
        round(col("Monthly_Balance"), 2)
    )

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
    "Customer_ID": StringType(),
    "Annual_Income": DoubleType(),
    "Monthly_Inhand_Salary": DoubleType(),
    "Num_Bank_Accounts": IntegerType(),
    "Num_Credit_Card": IntegerType(),
    "Interest_Rate": IntegerType(),
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
    "Loan_List": ArrayType(StringType()),
    "Loan": StringType()
}
    # column_type_map = {
    #     "Customer_ID": StringType(),
    #     "Annual_Income": DoubleType(),
    #     "Monthly_Inhand_Salary": DoubleType(),
    #     "Num_Bank_Accounts": IntegerType(),
    #     "Num_Credit_Card": IntegerType(),
    #     "Interest_Rate": IntegerType(),
    #     "Type_of_Loan": StringType(),
    #     "Delay_from_due_date": IntegerType(),
    #     "Num_of_Delayed_Payment": IntegerType(),
    #     "Changed_Credit_Limit": DoubleType(),
    #     "Num_Credit_Inquiries": IntegerType(),
    #     "Credit_Mix": StringType(),
    #     "Outstanding_Debt": DoubleType(),
    #     "Credit_Utilization_Ratio": DoubleType(),
    #     "Credit_History_Age": DoubleType(),
    #     "Payment_of_Min_Amount": StringType(),
    #     "Total_EMI_per_month": DoubleType(),
    #     "Amount_invested_monthly": DoubleType(),
    #     "Payment_Behaviour": StringType(),
    #     "Monthly_Balance": DoubleType(),      
    #     "snapshot_date": DateType(),
    # }

    #change to new dtype
    for column, new_type in column_type_map.items():
        df = df_cleaned.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_financials_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df