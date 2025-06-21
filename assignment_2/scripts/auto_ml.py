import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window

from pyspark.sql.functions import col, last, first, coalesce
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# to call this script: python auto_ml.py --snapshotdate "2024-09-01"
def is_valid_df(df):
    return df is not None and not df.rdd.isEmpty()

def main(snapshotdate):
    try:
        print('\n\n---starting job---\n\n')
        
        # Initialize SparkSession
        spark = pyspark.sql.SparkSession.builder \
            .appName("dev") \
            .master("local[*]") \
            .getOrCreate()
        
        # Set log level to ERROR to hide warnings
        spark.sparkContext.setLogLevel("ERROR")

        # get training data
        label_df =  None
        cust_fin_risk_df = None
        eng_df = None

    
        # --- load prediction store ---
        label_directory = f"./datamart/gold/label_store/"

        # Define date range
        start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
        end_date = datetime.strptime(snapshotdate, "%Y-%m-%d")

        # Generate monthly file dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y_%m_%d"))
            # Advance to first day of next month
            next_month = current.replace(day=28) + timedelta(days=4)
            current = next_month.replace(day=1)

        
        parquet_files = []

        for dt in dates:
            partition_name = 'gold_label_store_' + f'{dt}' + '.parquet'
            filepath = label_directory + partition_name
            print(filepath)
            parquet_files.append(filepath)

        existing_files = [f for f in parquet_files if os.path.exists(f)]

        # Read into PySpark DataFrame
        if existing_files:
            label_df = spark.read.parquet(*existing_files)
        else:
            print("No matching parquet files found in date range.")
        

        if label_df.count() >0:
            #get cust_fin_risk features
            cust_fin_risk_directory = f"./datamart/gold/feature_store/cust_fin_risk/"

            cust_fin_risk_parquet_files = []

            for dt in dates:
                partition_name = 'gold_ft_store_cust_fin_risk_' + f'{dt}' + '.parquet'
                filepath = cust_fin_risk_directory + partition_name
                print(filepath)
                cust_fin_risk_parquet_files.append(filepath)

            cust_fin_risk_existing_files = [f for f in cust_fin_risk_parquet_files if os.path.exists(f)]

            # Read into PySpark DataFrame
            if cust_fin_risk_existing_files:
                cust_fin_risk_df = spark.read.parquet(*cust_fin_risk_existing_files)
            else:
                print("No matching cust_fin_risk parquet files found in date range.")
            
            #get eng features
            eng_directory = f"./datamart/gold/feature_store/eng/"

            eng_parquet_files = []

            for dt in dates:
                partition_name = 'gold_ft_store_engagement_' + f'{dt}' + '.parquet'
                filepath = eng_directory + partition_name
                print(filepath)
                eng_parquet_files.append(filepath)

            eng_existing_files = [f for f in eng_parquet_files if os.path.exists(f)]

            # Read into PySpark DataFrame
            if eng_existing_files:
                eng_df = spark.read.parquet(*eng_existing_files)
            else:
                print("No matching eng parquet files found in date range.")

        else:
            print('no training labels')


        

        if all(is_valid_df(df) for df in [label_df, cust_fin_risk_df, eng_df]):
            features_sdf = eng_df.join(cust_fin_risk_df, on=["Customer_ID", "snapshot_date"], how="outer")

            # Define columns to fill
            fill_cols = [
                "click_1m", "click_2m", "click_3m", "click_4m", "click_5m", "click_6m",
                "Credit_History_Age", "Num_Fin_Pdts", "EMI_to_Salary", "Debt_to_Salary",
                "Repayment_Ability", "Loans_per_Credit_Item", "Loan_Extent", "Outstanding_Debt",
                "Interest_Rate", "Delay_from_due_date", "Changed_Credit_Limit"
            ]

            # Window for forward fill
            fwd_window = Window.partitionBy("Customer_ID").orderBy("snapshot_date").rowsBetween(Window.unboundedPreceding, 0)

            # Window for backward fill
            bwd_window = Window.partitionBy("Customer_ID").orderBy("snapshot_date").rowsBetween(0, Window.unboundedFollowing)

            # Apply fills
            df_filled = features_sdf
            for col_name in fill_cols:
                fwd_fill = last(col(col_name), ignorenulls=True).over(fwd_window)
                bwd_fill = first(col(col_name), ignorenulls=True).over(bwd_window)
                df_filled = df_filled.withColumn(col_name, coalesce(fwd_fill, bwd_fill))

            data_pdf = label_df.join(df_filled, on=["Customer_ID", "snapshot_date"], how="left").toPandas()
            
            # convert to pandas df and drop unecessary cols
            data_pdf = data_pdf[data_pdf["label"].notna()]
            data_pdf = data_pdf.dropna()

            # Make sure snapshot_date is datetime
            data_pdf["snapshot_date"] = pd.to_datetime(data_pdf["snapshot_date"])


            data_pdf["snapshot_date"] = pd.to_datetime(data_pdf["snapshot_date"])
            latest_date = data_pdf["snapshot_date"].max()

            # split data into train - test - oot
            oot_pdf = data_pdf[data_pdf["snapshot_date"] == latest_date]
            train_test_pdf = data_pdf[data_pdf["snapshot_date"] != latest_date]


            feature_cols = ['click_1m', 'click_2m', 'click_3m', 'click_4m', 'click_5m', 'click_6m',
                'Credit_History_Age', 'Num_Fin_Pdts', 'EMI_to_Salary', 'Debt_to_Salary',
                'Repayment_Ability', 'Loans_per_Credit_Item', 'Loan_Extent',
                'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date',
                'Changed_Credit_Limit']

            X_oot = oot_pdf[feature_cols]
            y_oot = oot_pdf["label"]
            X_train, X_test, y_train, y_test = train_test_split(
                train_test_pdf[feature_cols], train_test_pdf["label"], 
                test_size= 0.2,
                random_state=88,     # Ensures reproducibility
                shuffle=True,        # Shuffle the data before splitting
                stratify=train_test_pdf["label"]           # Stratify based on the label column
            )


            scaler = StandardScaler()

            transformer_stdscaler = scaler.fit(X_train) # Q which should we use? train? test? oot? all?

            # transform data
            X_train_processed = transformer_stdscaler.transform(X_train)
            X_test_processed = transformer_stdscaler.transform(X_test)
            X_oot_processed = transformer_stdscaler.transform(X_oot)

            print('start training xgb')
            # Define the XGBoost classifier
            xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

            # Define the hyperparameter space to search
            param_dist = {
                'n_estimators': [25, 50, 100],
                'max_depth': [2, 3, 5, 7],  # lower max_depth to simplify the model
                'learning_rate': [0.01, 0.1],
                'subsample': [0.6, 0.8, 1],
                'colsample_bytree': [0.6, 0.8, 1],
                'gamma': [0, 0.1],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2, 3]
            }

            # Create a scorer based on AUC score
            auc_scorer = make_scorer(roc_auc_score)

            # Set up the random search with cross-validation
            random_search = RandomizedSearchCV(
                estimator=xgb_clf,
                param_distributions=param_dist,
                scoring=auc_scorer,
                n_iter=100,  # Number of iterations for random search
                cv=3,       # Number of folds in cross-validation
                verbose=1,
                random_state=42,
                n_jobs=-1   # Use all available cores
            )

            # Perform the random search
            random_search.fit(X_train_processed, y_train)

            # Output the best parameters and best score
            print("Best parameters found: ", random_search.best_params_)
            print("Best AUC score: ", random_search.best_score_)

            # Evaluate the model on the train set
            best_model = random_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
            train_auc_score = roc_auc_score(y_train, y_pred_proba)
            print("Train AUC score: ", train_auc_score)

            # Evaluate the model on the test set
            best_model = random_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
            test_auc_score = roc_auc_score(y_test, y_pred_proba)
            print("Test AUC score: ", test_auc_score)

            # Evaluate the model on the oot set
            best_model = random_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
            oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
            print("OOT AUC score: ", oot_auc_score)


            model_artefact = {}

            model_artefact['model'] = best_model
            model_artefact['model_version'] = "credit_model_"+ f'{snapshotdate}'.replace('-','_')
            model_artefact['preprocessing_transformers'] = {}
            model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
            model_artefact['data_dates'] = dates
            model_artefact['data_stats'] = {}
            model_artefact['data_stats']['X_train'] = X_train.shape[0]
            model_artefact['data_stats']['X_test'] = X_test.shape[0]
            model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
            model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
            model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
            model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
            model_artefact['results'] = {}
            model_artefact['results']['auc_train'] = train_auc_score
            model_artefact['results']['auc_test'] = test_auc_score
            model_artefact['results']['auc_oot'] = oot_auc_score
            model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
            model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
            model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
            model_artefact['hp_params'] = random_search.best_params_


            # create model_bank dir
            model_bank_directory = "auto_model_bank/"

            if not os.path.exists(model_bank_directory):
                os.makedirs(model_bank_directory)

            # Full path to the file
            file_path = os.path.join(model_bank_directory, f'{snapshotdate}'.replace('-','_') + '.pkl')

            # Write the model to a pickle file
            with open(file_path, 'wb') as file:
                pickle.dump(model_artefact, file)

            print(f"Model saved to {file_path}")

            pc_monitor_directory = f"./datamart/gold/monitor/automl/"
            
            if not os.path.exists(pc_monitor_directory):
                os.makedirs(pc_monitor_directory)

            # save psi and auc into monitor directory
            pc_df = pd.DataFrame({'snapshotdate': [snapshotdate], 'auc_train': [train_auc_score], 'auc_test':[test_auc_score], 'auc_oot':[oot_auc_score]})

            partition_name = "performance_ceiling_" + 'xgb_'+ snapshotdate.replace('-','_') + '.csv'
            filepath = pc_monitor_directory + partition_name
            pc_df.to_csv(filepath)

            # Plot performance_ceiling trend
            # Get list of all CSV files in the folder
            csv_files = glob.glob(os.path.join(pc_monitor_directory, '*.csv'))

            # Read and combine all CSV files into one DataFrame
            df_combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            # Convert to datetime (if not already)
            df_combined['snapshotdate'] = pd.to_datetime(df_combined['snapshotdate'])
            # Sort by snapshotdate
            df_combined = df_combined.sort_values(by='snapshotdate')
            # reset index after sorting
            df_combined = df_combined.reset_index(drop=True)

            if len(df_combined['auc_test']) >0:
                plt.plot(df_combined['snapshotdate'], df_combined['auc_test'])
                plt.title('test_auc_score_across_snapshotdate_'+ f'{snapshotdate}'.replace('-','_'))
                plt.xticks(rotation=90)
                plt.xlabel('snapshotdate')
                plt.ylabel('test_auc_score')

            partition_name = 'test_auc_score_' + f'{snapshotdate}'.replace('-','_') + '.png'
            filepath = pc_monitor_directory + partition_name
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
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
