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
from pyspark.sql.functions import col, row_number, to_date, count, percentile_approx, desc, udf, collect_set, array_contains, lit, last
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType, ArrayType, NumericType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import regexp_extract, regexp_replace, when, trim, initcap, round, lower, split, explode, array_distinct, array_sort, concat_ws, expr

def feature_store_silver_table(snapshot_date_str, bronze_feature_store_directory, silver_feature_store_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_store_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_feature_store_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    features_df = df
    # clean clickstream
    # List of columns to impute
    columns_to_impute = [f"fe_{i}" for i in range(1, 21)]

    # Compute medians for each column
    medians = {
        col_name: features_df.approxQuantile(col_name, [0.5], 0.01)[0]
        for col_name in columns_to_impute
    }

    # Apply median imputation
    for col_name in columns_to_impute:
        median_value = medians[col_name]
        features_df = features_df.withColumn(
            col_name,
            when(col(col_name).isNull(), median_value).otherwise(col(col_name))
        )


    df_cleaned = features_df

    # 1. Clean Age: Extract digits and cast to Integer
    df_cleaned = df_cleaned.withColumn(
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
        .otherwise(initcap(regexp_replace(col("Occupation"), "_", " ")))  # Optional: beautify names
    )

    # Define the window partitioned by customer_id and ordered by snapshot_date
    w = Window.partitionBy("customer_id").orderBy("snapshot_date").rowsBetween(Window.unboundedPreceding, 0)

    # Apply last() with ignoreNulls=True to forward-fill values
    df_cleaned = df_cleaned.withColumn("Age", last("Age", ignorenulls=True).over(w)) \
                .withColumn("SSN", last("SSN", ignorenulls=True).over(w)) \
                .withColumn("Occupation", last("Occupation", ignorenulls=True).over(w))
    

    # Impute missing values with 'unknown' (cast numeric to string if needed)
    df_cleaned = df_cleaned.fillna({'SSN': 'Unknown', 'Occupation': 'Unknown'})

    # Compute median of Age (excluding nulls)
    median_age = df_cleaned.approxQuantile("Age", [0.5], 0.01)[0]

    df_cleaned = df_cleaned.fillna({"Age": median_age})

    # Ensure snapshot_date is timestamp (if it's not already)
    df_cleaned = df_cleaned.withColumn("snapshot_date", F.col("snapshot_date").cast("timestamp"))

    # Define a window to get the latest non-null Name per Customer_ID
    w = Window.partitionBy("Customer_ID").orderBy(F.col("snapshot_date").desc())

    # Get latest non-null Name
    df_cleaned = df_cleaned.withColumn(
        "latest_name",
        F.first("Name", ignorenulls=True).over(w)
    )

    # Fill Name column with latest if null, otherwise keep original
    df_cleaned = df_cleaned.withColumn(
        "Name_filled",
        F.when(F.col("Name").isNull(), F.col("latest_name")).otherwise(F.col("Name"))
    )

    # Fill remaining nulls (if customer had no name at all) with 'Unknown'
    df_cleaned = df_cleaned.withColumn(
        "Name_filled",
        F.when(F.col("Name_filled").isNull(), F.lit("Unknown")).otherwise(F.col("Name_filled"))
    )

    # Replace original Name column
    df_cleaned = df_cleaned.drop("Name", "latest_name").withColumnRenamed("Name_filled", "Name")


    # Clean annual income
    # Remove non-numeric characters
    df_cleaned = df_cleaned.withColumn(
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

    # forward fill
    categorical_cols = ['Type_of_Loan']

    # Define window spec for forward fill
    w = Window.partitionBy("customer_id").orderBy("snapshot_date").rowsBetween(Window.unboundedPreceding, 0)

    # Forward fill each numerical column
    for col_name in categorical_cols:
        df_cleaned = df_cleaned.withColumn(col_name, last(col_name, ignorenulls=True).over(w))


    # Clean up type of loan and split them nicely
    # 1. Fill null or empty strings with "Not_Specified"
    df_cleaned = df_cleaned.fillna("Not Specified", subset=["Type_of_Loan"])

    # Replace empty strings in 'Type_of_Loan' with 'Not Specified'
    df_cleaned = df_cleaned.withColumn(
        "Type_of_Loan",
        when(trim(col("Type_of_Loan")) == "", "Not Specified").otherwise(col("Type_of_Loan"))
    )

    # Step 1: Clean and standardize the loan types
    df_cleaned = df_cleaned.withColumn(
        "cleaned_loans",
        F.regexp_replace(
            F.lower(F.col("Type_of_Loan")),
            "(\s+and\s+)|(,\s+and\s+)|(,\s*)",  # Standardize all separators
            ","  # Replace with simple comma
        )
    )

    # Step 2: Split into array and trim whitespace
    df_cleaned = df_cleaned.withColumn(
        "loan_array",
        F.expr("transform(split(cleaned_loans, ','), x -> trim(x))")
    )


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

    #forwardd fill the numerical columns
    numerical_cols = [
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Credit_History_Age',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance'
    ]

    # Define window spec for forward fill
    w = Window.partitionBy("customer_id").orderBy("snapshot_date").rowsBetween(Window.unboundedPreceding, 0)

    # Forward fill each numerical column
    for col_name in numerical_cols:
        df_cleaned = df_cleaned.withColumn(col_name, last(col_name, ignorenulls=True).over(w))


    # Columns to impute
    cols_to_impute = [
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Credit_History_Age',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance'
    ]

    # Calculate medians using approxQuantile
    medians = {
        col_name: df_cleaned.approxQuantile(col_name, [0.5], 0.01)[0]
        for col_name in cols_to_impute
    }

    # Impute missing values with median
    for col_name in cols_to_impute:
        median_value = medians[col_name]
        df_cleaned = df_cleaned.withColumn(
            col_name,
            when(col(col_name).isNull(), median_value).otherwise(col(col_name))
        )

    # forward fill
    categorical_cols = ['Credit_Mix', 'Payment_Behaviour']

    # Define window spec for forward fill
    w = Window.partitionBy("customer_id").orderBy("snapshot_date").rowsBetween(Window.unboundedPreceding, 0)

    # Forward fill each numerical column
    for col_name in categorical_cols:
        df_cleaned = df_cleaned.withColumn(col_name, last(col_name, ignorenulls=True).over(w))


    # Categorical columns to impute with "Unknown"
    categorical_cols = ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    # Impute missing (null) values with "Unknown"
    for col_name in categorical_cols:
        df_cleaned = df_cleaned.withColumn(
            col_name,
            when(col(col_name).isNull(), "Unknown").otherwise(col(col_name))
        )


    # Cast Data types
    column_type_map = {
    'Customer_ID': StringType(),
    'snapshot_date': DateType(),
    'fe_1': DoubleType(),
    'fe_2': DoubleType(),
    'fe_3': DoubleType(),
    'fe_4': DoubleType(),
    'fe_5': DoubleType(),
    'fe_6': DoubleType(),
    'fe_7': DoubleType(),
    'fe_8': DoubleType(),
    'fe_9': DoubleType(),
    'fe_10': DoubleType(),
    'fe_11': DoubleType(),
    'fe_12': DoubleType(),
    'fe_13': DoubleType(),
    'fe_14': DoubleType(),
    'fe_15': DoubleType(),
    'fe_16': DoubleType(),
    'fe_17': DoubleType(),
    'fe_18': DoubleType(),
    'fe_19': DoubleType(),
    'fe_20': DoubleType(),
    'Age': IntegerType(),
    'SSN': StringType(),
    'Occupation': StringType(),
    'Annual_Income': DoubleType(),
    'Monthly_Inhand_Salary': DoubleType(),
    'Num_Bank_Accounts': IntegerType(),
    'Num_Credit_Card': IntegerType(),
    'Interest_Rate': IntegerType(),
    'Num_of_Loan': IntegerType(),
    'Type_of_Loan': StringType(),
    'Delay_from_due_date': IntegerType(),
    'Num_of_Delayed_Payment': IntegerType(),
    'Changed_Credit_Limit': DoubleType(),
    'Num_Credit_Inquiries': IntegerType(),
    'Credit_Mix': StringType(),
    'Outstanding_Debt': DoubleType(),
    'Credit_Utilization_Ratio': DoubleType(),
    'Credit_History_Age': DoubleType(),
    'Payment_of_Min_Amount': StringType(),
    'Total_EMI_per_month': DoubleType(),
    'Amount_invested_monthly': DoubleType(),
    'Payment_Behaviour': StringType(),
    'Monthly_Balance': DoubleType(),
    'Name': StringType(),
    'cleaned_loans':StringType(),
    'loan_array': ArrayType(StringType())}

    # Assuming df_cleaned is your input DataFrame
    df_casted = df_cleaned
    for column, new_type in column_type_map.items():
        if column in df_casted.columns:
            df_casted = df_casted.withColumn(column, col(column).cast(new_type))

    df = df_casted 

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_store_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath, 'row count:', df.count())
    
    return df