import ast
# import training
# import scoring
# import deployment
# import diagnostics
# import reporting
import json
from pathlib import Path
import os

import ingestion

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
input_folder_path = os.path.join(config['input_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Check and read new data
with open(f"{prod_deployment_path}/ingestedfiles.txt", 'r') as ingest_log:
    ingest_log = ingest_log.read().splitlines()

ingested_files = ["dataset" + line.split("dataset")[-1] for line in ingest_log if "dataset" in line]

print(ingested_files)

new_files = list(Path(input_folder_path).rglob('*.csv'))
new_files_names = [file.name for file in new_files if file.name not in ingested_files]
print(new_files_names)

if len(new_files) == 0:
    # logger.info("No new data files. Stop pipeline")
    exit()

# logger.info("New datasets found. Ingesting...")
ingestion.merge_multiple_dataframe()
# logger.info("New datasets ingested.")




# first, read ingestedfiles.txt

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt


##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
