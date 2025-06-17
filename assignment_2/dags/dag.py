from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 
 
    # --- feature store ---
    dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")

    dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_bronze_2")

    dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_bronze_3")

    bronze_clickstream_table = BashOperator(
        task_id='run_bronze_clickstream_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_clickstream_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    bronze_financial_table = BashOperator(
        task_id='run_bronze_financial_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_financial_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_attribute_table = BashOperator(
        task_id='run_bronze_attribute_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_attribute_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_clickstream_table = BashOperator(
        task_id='run_silver_clickstream_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_clickstream_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    silver_financial_table = BashOperator(
        task_id='run_silver_financial_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_financial_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_attribute_table = BashOperator(
        task_id='run_silver_attribute_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_attribute_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_cust_risk_store = BashOperator(
        task_id='run_gold_cust_risk_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_cust_risk_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    gold_engag_store = BashOperator(
        task_id='run_gold_engag_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_engag_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_data_bronze_1 >> bronze_clickstream_table >> silver_clickstream_table >> gold_engag_store
    dep_check_source_data_bronze_2 >> bronze_financial_table >> silver_financial_table >> gold_cust_risk_store
    dep_check_source_data_bronze_3 >> bronze_attribute_table >> silver_attribute_table
    gold_engag_store >> feature_store_completed
    gold_cust_risk_store >> feature_store_completed


    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_1_inference = BashOperator(
        task_id='model_1_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_2024_09_01.pkl"'
        ),
    )


    # model_2_inference = DummyOperator(task_id="model_2_inference")

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    # model_inference_start >> model_2_inference >> model_inference_completed


    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = BashOperator(
        task_id='model_1_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_monitor.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_2024_09_01.pkl"'
        ),
    )
    
    # model_2_monitor = DummyOperator(task_id="model_2_monitor")

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    # model_monitor_start >> model_2_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_1_automl = BashOperator(
        task_id='model_1_automl',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 auto_ml.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # model_2_automl = DummyOperator(task_id="model_2_automl")

    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    # model_automl_start >> model_2_automl >> model_automl_completed