import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
output_model_path = Path(config['output_model_path'])


def score_model() -> None:
    """
    Function for reporting. Calculates a confusion matrix using the test data and the deployed model.
    Writes the confusion matrix to the workspace.
    Returns
    -------

    """
    lr_model_fit = joblib.load(output_model_path / 'trainedmodel.pkl')
    test_set = pd.read_csv(test_data_path / 'testdata.csv')

    x_features = test_set.loc[:, ['number_of_employees', 'lastyear_activity', 'lastmonth_activity']].values.reshape(-1,
                                                                                                                     3)
    y_target = test_set['exited'].values.reshape(-1, 1).ravel()

    y_predictions = lr_model_fit.predict(x_features)

    cf_matrix = confusion_matrix(y_target, y_predictions)

    # To create the confussion matrix with nice formatting and coloring, I followed:
    # https://stackoverflow.com/questions/64800003/seaborn-confusion-matrix-heatmap-2-color-schemes-correct-diagonal-vs-wrong-re
    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)

    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
    sns.heatmap(cf_matrix, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))

    plt.savefig('confusionmatrix.png', bbox_inches='tight')


if __name__ == '__main__':
    score_model()
