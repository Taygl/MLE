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

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

import utils.feature_store_bronze_table
import utils.feature_store_silver_table
import utils.feature_store_gold_table

# import utils.clickstream_bronze_table
# import utils.clickstream_silver_table
# import utils.clickstream_gold_table

# import utils.features_attributes_bronze_table
# import utils.features_attributes_silver_table
# import utils.features_attributes_gold_table

# import utils.features_financials_bronze_table
# import utils.features_financials_silver_table
# import utils.features_financials_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create datalake 
print('creating datalake')
# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)


# create bronze datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)


# create bronze datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("gold label store row_count:",df.count())


# create bronze datalake
bronze_feature_store_directory = "datamart/bronze/feature_store/"

if not os.path.exists(bronze_feature_store_directory):
    os.makedirs(bronze_feature_store_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.feature_store_bronze_table.feature_store_bronze_table(date_str, bronze_feature_store_directory, spark)

# create silver datalake
silver_feature_store_directory = "datamart/silver/feature_store/"

if not os.path.exists(silver_feature_store_directory):
    os.makedirs(silver_feature_store_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.feature_store_silver_table.feature_store_silver_table(date_str, bronze_feature_store_directory, silver_feature_store_directory, spark)

# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.feature_store_gold_table.feature_store_gold_table(date_str, silver_feature_store_directory, gold_feature_store_directory, spark)


folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())


# # Clickstream
# print('Clickstream')
# # create bronze datalake
# bronze_clickstream_directory = "datamart/bronze/clickstream/"

# if not os.path.exists(bronze_clickstream_directory):
#     os.makedirs(bronze_clickstream_directory)

# # run bronze backfill
# for date_str in dates_str_lst:
#     utils.clickstream_bronze_table.feature_clickstream_bronze_table(date_str, bronze_clickstream_directory, spark)

# # create silver datalake
# silver_clickstream_directory = "datamart/silver/clickstream/"

# if not os.path.exists(silver_clickstream_directory):
#     os.makedirs(silver_clickstream_directory)

# # run silver backfill
# for date_str in dates_str_lst:
#     utils.clickstream_silver_table.feature_clickstream_silver_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)


# # create silver datalake
# gold_clickstream_directory = "datamart/gold/clickstream/"

# if not os.path.exists(gold_clickstream_directory):
#     os.makedirs(gold_clickstream_directory)

# # run gold backfill
# for date_str in dates_str_lst:
#     utils.clickstream_gold_table.feature_clickstream_gold_table(date_str, silver_clickstream_directory, gold_clickstream_directory, spark)


# folder_path = gold_clickstream_directory
# files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
# df = spark.read.option("header", "true").parquet(*files_list)
# print("clickstream row_count:",df.count())

# # features attributes
# print('features attributes')
# # create bronze datalake
# bronze_features_attributes_directory = "datamart/bronze/features_attributes/"

# if not os.path.exists(bronze_features_attributes_directory):
#     os.makedirs(bronze_features_attributes_directory)

# # run bronze backfill
# for date_str in dates_str_lst:
#     utils.features_attributes_bronze_table.features_attributes_bronze_table(date_str, bronze_features_attributes_directory, spark)

# # create silver datalake
# silver_features_attributes_directory = "datamart/silver/features_attributes/"

# if not os.path.exists(silver_features_attributes_directory):
#     os.makedirs(silver_features_attributes_directory)

# # run silver backfill
# for date_str in dates_str_lst:
#     utils.features_attributes_silver_table.features_attributes_silver_table(date_str, bronze_features_attributes_directory, silver_features_attributes_directory, spark)

# # create gold datalake
# gold_features_attributes_directory = "datamart/gold/features_attributes/"

# if not os.path.exists(gold_features_attributes_directory):
#     os.makedirs(gold_features_attributes_directory)

# # run gold backfill
# for date_str in dates_str_lst:
#     utils.features_attributes_gold_table.features_attributes_gold_table(date_str, silver_features_attributes_directory, gold_features_attributes_directory, spark)


# folder_path = gold_features_attributes_directory
# files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
# df = spark.read.option("header", "true").parquet(*files_list)
# print("features_attributes row_count:",df.count())


# # features financials
# print('features financials')
# # create bronze datalake
# bronze_features_financials_directory = "datamart/bronze/features_financials/"

# if not os.path.exists(bronze_features_financials_directory):
#     os.makedirs(bronze_features_financials_directory)

# # run bronze backfill
# for date_str in dates_str_lst:
#     utils.features_financials_bronze_table.features_financials_bronze_table(date_str, bronze_features_financials_directory, spark)

# # create silver datalake
# silver_features_financials_directory = "datamart/silver/features_financials/"

# if not os.path.exists(silver_features_financials_directory):
#     os.makedirs(silver_features_financials_directory)

# # run silver backfill
# for date_str in dates_str_lst:
#     utils.features_financials_silver_table.features_financials_silver_table(date_str, bronze_features_financials_directory, silver_features_financials_directory, spark)


# # create gold datalake
# gold_features_financials_directory = "datamart/gold/features_financials/"

# if not os.path.exists(gold_features_financials_directory):
#     os.makedirs(gold_features_financials_directory)

# # run gold backfill
# for date_str in dates_str_lst:
#     utils.features_financials_gold_table.features_financials_gold_table(date_str, silver_features_financials_directory, gold_features_financials_directory, spark)


# folder_path = gold_features_financials_directory
# files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
# df = spark.read.option("header", "true").parquet(*files_list)
# print("features_financials row_count:",df.count())