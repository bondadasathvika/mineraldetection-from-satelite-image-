import numpy as np


def minmax_normalize(spectrum):

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64
    )

    minimum = np.min(spectrum)
    maximum = np.max(spectrum)

    if maximum - minimum == 0:

        return np.zeros_like(spectrum)

    return (
        (spectrum - minimum)
        /
        (maximum - minimum)
    )


def l2_normalize(spectrum):

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64
    )

    norm = np.linalg.norm(spectrum)

    if norm == 0:

        return spectrum

    return spectrum / norm


def normalize_cube(cube):

    cube = np.asarray(
        cube,
        dtype=np.float64
    )

    norm = np.linalg.norm(
        cube,
        axis=-1,
        keepdims=True
    )

    norm[norm == 0] = 1

    return cube / norm