import os
import joblib
import pandas as pd
import subprocess
import timeit
from pathlib import Path
import json
from typing import Any, List
from ingestion import logger, FORMAT, formatter


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
prod_deployment_path = Path(config['prod_deployment_path'])
diagnosis_path = Path(config['diagnosis_path'])

diagnosis_path.mkdir(exist_ok=True)

diagnosis_log = logger('diagnosis', diagnosis_path / 'diagnosis_log.txt')


def model_predictions(data_path: Path) -> Any:
    """
    Function to get model predictions. Read the deployed model and a test dataset, calculate predictions
    Parameters
    ----------
    data_path

    Returns: A list containing all predictions
    -------

    """
    data_frame = pd.read_csv(data_path)
    lr_model_fit = joblib.load(prod_deployment_path / "trainedmodel.pkl")

    x_features = data_frame.loc[:, ['number_of_employees', 'lastyear_activity', 'lastmonth_activity']].values.reshape(-1, 3)
    y_target = data_frame['exited'].values.reshape(-1, 1).ravel()

    y_predictions = lr_model_fit.predict(x_features)

    diagnosis_log.info(f"y_preds : {y_predictions}")

    return y_predictions


def dataframe_summary() -> List[float]:
    """
    Function to get summary statistics
    Returns: list containing all summary statistics.
    -------

    """
    data = pd.read_csv(dataset_csv_path / "finaldata.csv")

    num_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited']

    mean_vals = [data[col].mean() for col in num_cols]
    median_vals = [data[col].median() for col in num_cols]
    std_vals =  [data[col].std() for col in num_cols]
    diagnosis_log.info(f"mean_vals : {mean_vals}")
    diagnosis_log.info(f"median_vals : {median_vals}")
    diagnosis_log.info(f"std_vals : {std_vals}")


    return [mean_vals, median_vals, std_vals]


def missing_data() -> float:
    """
    Function to check for NA values.
    Returns: Percentage of missing data.
    -------

    """
    data = pd.read_csv(dataset_csv_path / "finaldata.csv")
    na_count = [data[col].isna().sum() for col in data.columns]
    total_rows = len(data)
    missing_prcnt = [missing_vals / total_rows for missing_vals in na_count]

    diagnosis_log.info(f"missing_prcnt : {missing_prcnt}")


    return missing_prcnt


def execution_time() -> List[float]:
    """
    Function to calculate ingestion and training times.
    Returns: List of floats with ingestion and training times.
    -------

    """
    start_ingest = timeit.default_timer()
    os.system('python ingestion.py')
    ingest_timing = timeit.default_timer() - start_ingest

    start_train = timeit.default_timer()
    os.system('python training.py')
    train_timing = timeit.default_timer() - start_train

    diagnosis_log.info(f"ingest_timing : {ingest_timing}")
    diagnosis_log.info(f"train_timing : {train_timing}")

    return [ingest_timing, train_timing]


def outdated_packages_list() -> bytes:
    """
    Function to check dependencies
    Returns
    -------

    """
    pip_list_outdated = subprocess.check_output(['pip', 'list', '--outdated'])

    diagnosis_log.info(f"pip_list_outdated : {pip_list_outdated}")

    return pip_list_outdated


if __name__ == '__main__':
    model_predictions(dataset_csv_path / 'finaldata.csv')
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
