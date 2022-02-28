import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Union
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Function to read all csv files in the input_path and merge the data into a single dataframe.
    The data files should have the same schema.
    :param input_path: Path object from which the csv files will be ingested.
    :return: Pandas dataframe object.
    """
    csv_paths = Path(input_path).rglob('*.csv')
    data_frame_tot = pd.DataFrame()
    for path in csv_paths:
        data_frame_temp = pd.read_csv(path)
        data_frame_tot = pd.concat([data_frame_tot, data_frame_temp], ignore_index=True)

    return data_frame_tot


if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path)
