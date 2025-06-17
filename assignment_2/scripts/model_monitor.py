import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
print(sys.version)


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"



def calculate_psi(expected, actual, buckets=10):
    # Convert to numeric and drop non-convertible values
    expected = pd.to_numeric(expected, errors='coerce')
    actual = pd.to_numeric(actual, errors='coerce')

    # Drop NaNs after conversion
    expected = expected.dropna()
    actual = actual.dropna()

    # Create breakpoints from expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        raise ValueError("Not enough unique values in expected data to form bins.")

    # Bin data
    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

    # Get distributions
    expected_dist = pd.value_counts(expected_bins, normalize=True).sort_index()
    actual_dist = pd.value_counts(actual_bins, normalize=True).sort_index()

    # Align and add small epsilon to avoid division by zero
    expected_dist, actual_dist = expected_dist.align(actual_dist, fill_value=1e-6)

    # PSI calculation
    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()

    return psi

def main(snapshotdate, modelname):
    try:
        print('\n\n---starting job---\n\n')
        
        # Initialize SparkSession
        spark = pyspark.sql.SparkSession.builder \
            .appName("dev") \
            .master("local[*]") \
            .getOrCreate()
        
        # Set log level to ERROR to hide warnings
        spark.sparkContext.setLogLevel("ERROR")

        # check label drift using psi
    
        # --- load prediction store ---
        predictions_directory = f"./datamart/gold/model_predictions/{modelname[:-4]}/"

        partition_name = f'{modelname[:-4]}' + '_predictions_' + f'{snapshotdate}'.replace('-','_') + '.parquet'
        filepath = predictions_directory + partition_name
        predictions_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', predictions_df.count())

        predictions_df = predictions_df.toPandas()
        new_predict_arr = predictions_df["model_predictions"]

        #get last month's prediction
        date_obj = pd.to_datetime(snapshotdate)
        # Subtract 1 month
        previous_predict_date = (date_obj - pd.DateOffset(months=1)).strftime("%Y-%m-%d")

        partition_name = f'{modelname[:-4]}' + '_predictions_' + f'{previous_predict_date}'.replace('-','_') + '.parquet'
        filepath = predictions_directory + partition_name
        previous_predictions_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', predictions_df.count())

        previous_predictions_df = previous_predictions_df.toPandas()
        previous_predict_arr = previous_predictions_df["model_predictions"]
        
        prediction_psi = None
        #cal psi
        prediction_psi = calculate_psi(previous_predict_arr, new_predict_arr)

        print(prediction_psi)

        #send alert if huge drift
        if prediction_psi > 0.25:
            print(f'PSI: {prediction_psi}, major prediction labels shift detected')
        else:
            print(f'PSI: {prediction_psi}, prediction labels within range')


        monitor_directory = f"./datamart/gold/monitor/"

        if not os.path.exists(monitor_directory):
            os.makedirs(monitor_directory)

        # Plot KDE of predictions of previous vs current
        sns.kdeplot(previous_predict_arr, label='previous_prediction', shade=True)
        sns.kdeplot(new_predict_arr, label='current_prediction', shade=True)

        # Add legend and labels
        plt.legend()
        plt.title("KDE of predictions")
        plt.xlabel("Value")
        plt.ylabel("Density")

        partition_name = f'{modelname[:-4]}' + '_predictions_distribution_' + f'{snapshotdate}'.replace('-','_') + '.png'
        filepath = monitor_directory + partition_name

        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close('all')

        # monitor model accuracy
        label_directory = f"./datamart/gold/label_store/"

        partition_name = 'gold_label_store_' +  f'{snapshotdate}'.replace('-','_') + '.parquet'
        filepath = label_directory + partition_name
        label_df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', predictions_df.count())

        label_df = label_df.toPandas()

        print(predictions_df.head(20))

        print(label_df.head(20))

        # merged_df = label_df.merge(predictions_df, on="Customer_ID", how="left")

        merged_df = label_df
        merged_df['model_predictions'] = predictions_df['model_predictions']

        merged_df = merged_df[['label','model_predictions']]

        merged_df = merged_df.dropna()

        print(merged_df.head(20))

        roc_auc = None

        if len(merged_df['label']) > 0:
            roc_auc = roc_auc_score(merged_df['label'], merged_df["model_predictions"])
            # in reality send alert or retrain model
            if roc_auc< 0.65:
                print(f'roc_auc: {roc_auc}, retraining needed')
            else:
                print(f'roc_auc: {roc_auc}, model performance ok')
        else:
            print('labels not available!')


        # save psi and auc into monitor directory
        monitor_df = pd.DataFrame({'snapshotdate': [snapshotdate], 'modelname': [modelname], 'predict_psi':[prediction_psi], 'roc_auc':[roc_auc]})

        partition_name = "monitor_" + f'{modelname[:-4]}_'+ snapshotdate.replace('-','_') + '.csv'
        filepath = monitor_directory + partition_name
        monitor_df.to_csv(filepath)

        # Plot AUC trend
        # Get list of all CSV files in the folder
        csv_files = glob.glob(os.path.join(monitor_directory, '*.csv'))

        # Read and combine all CSV files into one DataFrame
        df_combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        # Convert to datetime (if not already)
        df_combined['snapshotdate'] = pd.to_datetime(df_combined['snapshotdate'])
        # Sort by snapshotdate
        df_combined = df_combined.sort_values(by='snapshotdate')
        # reset index after sorting
        df_combined = df_combined.reset_index(drop=True)

        if len(df_combined['predict_psi']) >0:
            plt.plot(df_combined['snapshotdate'], df_combined['predict_psi'])
            plt.title('PSI across snapshotdate')
            plt.xticks(rotation=90)
            plt.xlabel('snapshotdate')
            plt.ylabel('PSI')

        partition_name = 'psi_trend_' + f'{modelname[:-4]}_' + f'{snapshotdate}'.replace('-','_') + '.png'
        filepath = monitor_directory + partition_name
        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close('all')

        if len(df_combined['roc_auc']) >0:
            plt.plot(df_combined['snapshotdate'], df_combined['roc_auc'])
            plt.title('roc_auc across snapshotdate')
            plt.xticks(rotation=90)
            plt.xlabel('snapshotdate')
            plt.ylabel('roc_auc')


        partition_name = 'roc_auc_trend_' + f'{modelname[:-4]}_' + f'{snapshotdate}'.replace('-','_') + '.png'
        filepath = monitor_directory + partition_name
        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close('all')

        
        # --- end spark session --- 
        spark.stop()
        
        print('\n\n---completed job---\n\n')

    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
