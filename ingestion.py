"""
Author: Arturo Opsetmoen Amador.
Ingestion module.
"""


import pandas as pd
import logging
from pathlib import Path
from typing import Union
import json


FORMAT = '%(asctime)-15s %(message)s'
formatter = logging.Formatter(FORMAT)


def logger(name: str, log_file: str, level:logging.INFO=logging.INFO) -> logging.Logger:
    """

    Parameters
    ----------
    name
    log_file
    level

    Returns
    -------

    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger_ = logging.getLogger(name)
    logger_.setLevel(level)
    logger_.addHandler(handler)

    return logger_


ingestion_log = logger('ingestedfiles', 'ingestedfiles.txt')

with open('config.json', 'r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe(input_path: Union[str, Path], log_path: Union[str, Path]) -> pd.DataFrame:
    """
    Function to read all csv files in the input_path and merge the data into a single dataframe.
    The data files should have the same schema. The function also generates a log with all files
    read during the ingestion process.

    :param log_path: Path to write the ingestion logs to.
    :param input_path: Path object from which the csv files will be ingested.
    :return: Pandas dataframe object.
    """
    csv_paths = list(Path(input_path).rglob('*.csv'))

    ingestion_log.info(f"Files ingested from the folders: \n")
    data_frame_tot = pd.DataFrame()
    for path in csv_paths:
        ingestion_log.info(f"Files ingested from: {path}")
        data_frame_temp = pd.read_csv(path)
        ingestion_log.info(f"Number of rows ingested: {data_frame_temp.shape[0]}")
        data_frame_tot = pd.concat([data_frame_tot, data_frame_temp], ignore_index=True)


    num_cols = data_frame_tot.shape[1]
    num_rows = data_frame_tot.shape[0]
    ingestion_log.info(f"Merged dataframe cols: {num_cols}")
    ingestion_log.info(f"Merged dataframe rows: {num_rows}")




    return data_frame_tot


def drop_dups(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function to drop duplicates taking all columns into consideration. Default: keeps first.
    :param data_frame: Pandas dataframe
    :return: Pandas dataframe
    """
    data_frame = data_frame.drop_duplicates()
    return data_frame


def write_data_frame(data_frame: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Function to write the input dataframe to the output path.
    :param data_frame: Pandas dataframe
    :param output_path: String or Path object specifying the output path.
    :return: None
    """
    data_frame.to_csv(Path(output_path) / "finaldata.csv")


if __name__ == '__main__':
    data_frame = merge_multiple_dataframe(input_folder_path, output_folder_path)
    data_frame = drop_dups(data_frame)
    write_data_frame(data_frame, output_folder_path)
