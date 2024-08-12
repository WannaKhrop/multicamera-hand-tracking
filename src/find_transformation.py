"""
Module should be run if it's necessary to find a transformation for calibration data.

Author: Ivan Khrop
Date: 06.08.2024
"""
from utils.utils import umeyama, linear_transfomation

import argparse
import os
import glob
from pandas import read_csv
import numpy as np

from utils.constants import PATH_TO_TRANSFORMATION, PATH_TO_DATA_FOLDER, NUMPY_FILE_EXT


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Find transformation for point clouds."
    )

    # Add arguments
    parser.add_argument(
        "--mode", type=str, default="mse", help='Possible options are: "mse", "umeyama"'
    )

    # Parse the arguments
    args = parser.parse_args()

    # check arguments
    assert args.mode in ["mse", "umeyama"], 'Possible modes are: "mse", "umeyama"'

    # find all .csv files in data directory
    file_pattern = os.path.join(PATH_TO_DATA_FOLDER, "*.csv")

    # Find all CSV files in the directory
    csv_files = glob.glob(file_pattern)

    # Read each CSV file and store the DataFrames in a list
    for file in csv_files:
        # get camera id as string
        camera_id = file.split("\\")[-1].rstrip(".csv")

        # get points
        points = np.hsplit(read_csv(file).values, 2)
        camera_coords = points[0].T
        world_coords = points[1].T

        # find a transformation
        if args.mode == "mse":
            result = linear_transfomation(camera_coords, world_coords)
        else:
            result = umeyama(camera_coords.T, world_coords.T)

        # save the resulting matrix
        file_path = os.path.join(PATH_TO_TRANSFORMATION, camera_id + NUMPY_FILE_EXT)
        np.save(file=file_path, arr=result)

        print(f"Transformation for camera {camera_id} is saved.")


if __name__ == "__main__":
    main()
