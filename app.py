from flask import Flask, session, jsonify, request, Response
import pandas as pd
import numpy as np
import pickle
import scoring
from typing import Tuple
import diagnostics
import json
from pathlib import Path


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])

prediction_model = None


@app.route("/prediction", methods=['POST','OPTIONS'])
def predict() -> Tuple[Response, int]:
    """
    Prediction Endpoint. Calls the prediction function we created.
    Returns: return value for prediction outputs.
    -------

    """
    json_data = request.get_json()
    input_file = Path(json_data['input_file'])

    predics = diagnostics.model_predictions(input_file)

    return jsonify({"predictions": str(predics)}), 200


#######################
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats_model() -> Tuple[Response, int]:
    """
    Scoring Endpoint.Checks the score of the deployed model
    Returns
    -------

    """
    model_score = scoring.score_model()

    return jsonify({"score": model_score}), 200


#######################
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats_data() -> Tuple[Response, int]:
    """
    Summary Statistics Endpoint. Checks means, medians, and modes for each column
    Returns
    -------

    """
    summary_statistics = diagnostics.dataframe_summary()

    return jsonify({"summary_stats": summary_statistics}), 200


#######################
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats_diagnosis() -> Tuple[Response, int]:
    """
    Diagnostics Endpoint. Checks timing and percent NA values
    Returns: add return value for all diagnostics
    -------

    """
    ingest_time, training_time = diagnostics.execution_time()
    missing_statistics = diagnostics.missing_data()
    outdated_pip = diagnostics.outdated_packages_list()

    return jsonify({"ingest_time": str(ingest_time), "training_time": str(training_time),
                    "missing_statistics": str(missing_statistics),
                    "outdated_pip": str(outdated_pip)}), 200


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
