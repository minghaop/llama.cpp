# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json

import pandas as pd


def read_json(json_path: str) -> dict:
    """Reads a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(json_path) as f:
        data = json.load(f)
    return data


def dump_json(data: dict, json_path: str) -> None:
    """Dumps data into a JSON file.

    Args:
        data: Data to be dumped into JSON.
        json_path: Path to the JSON file.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def dump_csv(data_frame: dict, csv_path: str, index: bool = False) -> None:
    """Export dictionary data to CSV file with given path

    Args:
        data_frame (dict): Data to be dumped.
        csv_path (str): Path to output CSV file.
        index (bool): Whether to include index in the CSV file. Default is False
    """
    df = pd.DataFrame(data_frame)
    df.to_csv(csv_path, sep=",", index=index, header=True)
