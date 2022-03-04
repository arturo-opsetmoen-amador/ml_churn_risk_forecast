
import requests
from pathlib import Path
import json
from ingestion import logger
from typing import List


URL = "http://172.17.0.2:8000/"

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = Path(config['output_model_path'])
apicalls_log = logger('apicalls', output_model_path / 'apicalls_log.txt')


def apicalls() -> List[requests.Response]:
    """
    Function to call the API endpoints and generate a log.
    Returns List with API responses.
    -------

    """
    response1 = requests.post(
        f'{URL}prediction', json={
            "input_file": "testdata/testdata.csv"})
    response2 = requests.get(f'{URL}scoring')
    response3 = requests.get(f'{URL}summarystats')
    response4 = requests.get(f'{URL}diagnostics')

    # #combine all API responses
    responses = [response1, response2, response3, response4]

    apicalls_log.info(f"Prediction response code: {response1.status_code}. Prediction response text: {response1.text}")
    apicalls_log.info(f"Scoring response code: {response2.status_code}. Prediction response text: {response2.text}")
    apicalls_log.info(f"Summary stats response code: {response3.status_code}. Prediction response text: {response3.text}")
    apicalls_log.info(f"Diagnostics response code: {response4.status_code}. Prediction response text: {response4.text}")

    return responses


if __name__ == '__main__':
    apicalls()








