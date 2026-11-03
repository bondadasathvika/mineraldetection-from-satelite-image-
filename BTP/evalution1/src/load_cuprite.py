import os
import numpy as np
import scipy.io as sio


def load_mat_file(filepath):

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"File not found: {filepath}"
        )

    data = sio.loadmat(filepath)

    # Remove MATLAB metadata
    variables = {
        key: value
        for key, value in data.items()
        if not key.startswith("__")
    }

    return variables


def print_variables(data):

    print("\nMATLAB VARIABLES")
    print("-" * 60)

    for name, value in data.items():

        print(
            f"{name:30s} "
            f"shape={getattr(value, 'shape', None)} "
            f"dtype={getattr(value, 'dtype', None)}"
        )


def find_numeric_arrays(data):

    arrays = {}

    for name, value in data.items():

        if isinstance(value, np.ndarray):

            if np.issubdtype(
                value.dtype,
                np.number
            ):

                arrays[name] = value

    return arrays


def load_cuprite(
    data_path,
    gt_path
):

    data = load_mat_file(data_path)

    ground_truth = load_mat_file(gt_path)

    print("\n========== DATA FILE ==========")

    print_variables(data)

    print("\n========== GROUND TRUTH FILE ==========")

    print_variables(ground_truth)

    return data, ground_truth