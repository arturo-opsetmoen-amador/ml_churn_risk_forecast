from flask import Flask, session, jsonify, request
import pandas as pd
import joblib
from pathlib import Path
from sklearn import metrics
from ingestion import logger, FORMAT, formatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
output_model_path = Path(config['output_model_path'])

scoring_log = logger('latestscore', output_model_path / 'latestscore.txt')


def score_model() -> None:
    """
    This function takes a trained model, loads test data, and calculates an F1 score for the model relative
     to the test data.
    Returns
    -------

    """

    lr_model_fit = joblib.load(output_model_path / 'trainedmodel.pkl')

    test_set = pd.read_csv(test_data_path / "testdata.csv")

    x_features = test_set.loc[:, ['number_of_employees', 'lastyear_activity', 'lastmonth_activity']].values.reshape(-1,
                                                                                                                     3)
    y_target = test_set['exited'].values.reshape(-1, 1).ravel()

    y_predictions = lr_model_fit.predict(x_features)

    f1_score = metrics.f1_score(y_target, y_predictions)

    scoring_log.info(f"F1 score:  {f1_score}")

    return None


if __name__ == '__main__':
    score_model()
