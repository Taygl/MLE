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


def feature_store_bronze_table(snapshot_date_str, bronze_clickstream_directory, bronze_attribute_directory, bronze_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/feature_clickstream.csv"

    # load data - IRL ingest from back end source system
    clickstream_df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    clickstream_df.toPandas().to_csv(filepath, index=False)
    print('clickstream_df saved to:', filepath)


    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/features_attributes.csv"

    # load data - IRL ingest from back end source system
    attribute_df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_attribute_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attribute_directory + partition_name
    attribute_df.toPandas().to_csv(filepath, index=False)
    print('attribute_df saved to:', filepath)


    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/features_financials.csv"

    # load data - IRL ingest from back end source system
    financial_df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_financials_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    financial_df.toPandas().to_csv(filepath, index=False)
    print('financial_df saved to:', filepath)

    return clickstream_df, attribute_df, financial_df
