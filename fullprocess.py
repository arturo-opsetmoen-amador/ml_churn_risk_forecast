import ast
# import training
# import scoring
import deployment
import diagnostics
import reporting
import json
from pathlib import Path
import os
from ingestion import logger, FORMAT, formatter

import ingestion
import reporting
import scoring
import training

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
input_folder_path = os.path.join(config['input_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

logger = logger('full_process', f"{prod_deployment_path}/full_process_log.txt")

##################Check and read new data
# first, read ingestedfiles.txt

with open(f"{prod_deployment_path}/ingestedfiles.txt", 'r') as ingest_log:
    ingest_log = ingest_log.read().splitlines()

ingested_files = ["dataset" + line.split("dataset")[-1] for line in ingest_log if "dataset" in line]
logger.info(f"Ingested files: {ingested_files}")

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = list(Path(input_folder_path).rglob('*.csv'))
new_files_names = [file.name for file in new_files if file.name not in ingested_files]
logger.info(f"New files: {new_files_names}")

##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) == 0:
    logger.info("No new data files. Stop pipeline")
    exit()

logger.info("New datasets found. Ingesting...")
ingestion.run_ingestion()
logger.info("New datasets ingested.")

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
logger.info("Training new model after new data has been found.")

training.train_model()

logger.info("Model trained --> Scoring new model.")

new_model_score = scoring.score_model()

with open(f"{prod_deployment_path}/latestscore.txt", 'r') as old_score:
    old_model_score = old_score.read().splitlines()

news_hist_score = old_model_score[-1]
old_model_score = float(news_hist_score.split("F1 score:  ")[-1])

##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
logger.info(f"New model score: {new_model_score}, Old model score: {old_model_score}")

score_delta = new_model_score > old_model_score  # Boolean

if score_delta:
    logger.info("No performance driftfound... Stopping the pipeline.")
    exit()


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle()
logger.info(f"New model saved to pickle")

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
diagnostics.run_diagnostics()
reporting.score_model()
logger.info(f"New diagnostics generated. New scoring report generated. Deployment successful")





