import numpy as np


def spectral_angle(spectrum1, spectrum2):
    spectrum1 = np.asarray(spectrum1, dtype=np.float64)
    spectrum2 = np.asarray(spectrum2, dtype=np.float64)

    numerator = np.dot(spectrum1, spectrum2)

    denominator = (
        np.linalg.norm(spectrum1)
        * np.linalg.norm(spectrum2)
    )

    if denominator == 0:
        return np.pi / 2

    cosine = numerator / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.arccos(cosine)


def sam_map(cube, reference):
    height, width, bands = cube.shape

    result = np.zeros((height, width), dtype=np.float64)

    for row in range(height):
        for col in range(width):
            pixel = cube[row, col, :]
            result[row, col] = spectral_angle(
                pixel,
                reference
            )

    return result


def fast_sam_map(cube, reference):
    cube = np.asarray(cube, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    reference_norm = np.linalg.norm(reference)

    if reference_norm == 0:
        raise ValueError("Reference spectrum has zero norm.")

    cube_norm = np.linalg.norm(cube, axis=2)

    cube_norm[cube_norm == 0] = 1e-12

    dot_product = np.sum(
        cube * reference,
        axis=2
    )

    cosine = dot_product / (
        cube_norm * reference_norm
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.arccos(cosine)