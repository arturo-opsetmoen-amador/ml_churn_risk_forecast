from flask import Flask, session, jsonify, request
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
import json

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path'])


def train_model() -> None:
    finaldata_path = dataset_csv_path / "finaldata.csv"
    finaldata = pd.read_csv(finaldata_path)
    model_path.mkdir(exist_ok=True)
    x_features = finaldata.loc[:, ['number_of_employees', 'lastyear_activity', 'lastmonth_activity']].values.reshape(-1,
                                                                                                                     3)
    y_target = finaldata['exited'].values.reshape(-1, 1).ravel()

    log_reg_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='ovr', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

    lr_model_fit = log_reg_model.fit(x_features, y_target)

    joblib.dump(lr_model_fit, model_path / 'trainedmodel.pkl')

    return None


if __name__ == '__main__':
    train_model()
