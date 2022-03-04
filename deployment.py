from flask import Flask, session, jsonify, request
import pandas as pd
import shutil
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
prod_deployment_path = Path(config['prod_deployment_path'])
output_model_path = Path(config['output_model_path'])

prod_deployment_path.mkdir(exist_ok=True)


def store_model_into_pickle() -> None:
    """

    Parameters
    ----------
    model

    Returns
    -------

    """
    lr_model_fit = output_model_path / "trainedmodel.pkl"
    deploy_model_path = prod_deployment_path / "trainedmodel.pkl"
    shutil.copy2(src=lr_model_fit, dst=deploy_model_path)

    score_path = output_model_path / "latestscore.txt"
    deploy_score_path = prod_deployment_path / "latestscore.txt"
    shutil.copy2(src=score_path, dst=deploy_score_path)

    ingest_path = dataset_csv_path / "ingestedfiles.txt"
    deploy_ingest_path = prod_deployment_path / "ingestedfiles.txt"
    shutil.copy2(src=ingest_path, dst=deploy_ingest_path)

    return None


if __name__ == "__main__":
    store_model_into_pickle()


        

