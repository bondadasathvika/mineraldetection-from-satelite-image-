import os
import sys

import numpy as np

from src.load_cuprite import (
    load_cuprite
)

from src.sam import (
    fast_sam_map
)

from src.detection import (
    detect_from_sam,
    calculate_detection_percentage
)

from src.visualization import (
    save_sam_map,
    save_detection_map,
    save_binary_map,
    save_detection_overlay,
    plot_spectrum,
    plot_two_spectra
)


# ============================================================
# PATHS
# ============================================================

DATA_FILE = (
    "data/Cuprite/"
    "Cuprite_data_R188.mat"
)

GT_FILE = (
    "data/Cuprite/"
    "Cuprite_GT_nEnd12.mat"
)

OUTPUT_DIR = "outputs"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_MINERAL = "Alunite"

# SAM threshold in radians.
#
# IMPORTANT:
# This is an initial threshold.
# Later we will determine it more carefully.
#
SAM_THRESHOLD = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HYPERSPECTRAL MINERAL DETECTION")
    print("=" * 70)

    print("\nTarget mineral:", TARGET_MINERAL)

    # --------------------------------------------------------
    # STEP 1
    # Load MATLAB files
    # --------------------------------------------------------

    print("\n[1] Loading Cuprite data...")

    data, ground_truth = load_cuprite(
        DATA_FILE,
        GT_FILE
    )

    # --------------------------------------------------------
    # STEP 2
    # Print available variables
    # --------------------------------------------------------

    print("\n[2] Available image variables:")

    for name, value in data.items():

        print(
            name,
            getattr(value, "shape", None)
        )

    print("\n[3] Available ground-truth variables:")

    for name, value in ground_truth.items():

        print(
            name,
            getattr(value, "shape", None)
        )

    print("\n[4] Evaluation 1: Alunite SAM detection")
    print("=" * 70)

    nRow = int(data["nRow"][0, 0])
    nCol = int(data["nCol"][0, 0])
    nBand = int(data["nBand"][0, 0])

    if nRow != 250 or nCol != 190 or nBand != 188:
        raise ValueError(
            "Unexpected image dimensions. Expected nRow=250, "
            "nCol=190, nBand=188."
        )

    image_data = np.asarray(data["Y"], dtype=np.float64)

    if image_data.ndim != 2 or image_data.shape[0] != 188:
        raise ValueError(
            f"Y must have 188 spectral bands in shape (188, pixels); "
            f"received {image_data.shape}."
        )

    expected_pixels = nRow * nCol

    if image_data.shape[1] != expected_pixels:
        raise ValueError(
            f"Y cannot reshape to {nRow} x {nCol} x {nBand}: "
            f"expected {expected_pixels} pixels, "
            f"received {image_data.shape[1]}."
        )

    image_cube = image_data.T.reshape(
        nRow,
        nCol,
        nBand
    )

    endmembers = np.asarray(ground_truth["M"], dtype=np.float64)
    alunite_spectrum = endmembers[:, 0]

    if alunite_spectrum.shape != (224,):
        raise ValueError(
            "The Alunite endmember must have shape (224,) before "
            f"band selection; received {alunite_spectrum.shape}."
        )

    selected_bands = np.asarray(
        ground_truth["slctBnds"]
    ).reshape(-1).astype(np.int64)

    if selected_bands.size != 188:
        raise ValueError(
            "slctBnds must contain exactly 188 band indices; "
            f"received {selected_bands.size}."
        )

    if (
        selected_bands.min() >= 1
        and selected_bands.max() <= endmembers.shape[0]
    ):
        selected_band_indices = selected_bands - 1
        print("slctBnds indexing: MATLAB 1-based; converted to Python 0-based.")
    elif (
        selected_bands.min() >= 0
        and selected_bands.max() < endmembers.shape[0]
    ):
        selected_band_indices = selected_bands
        print("slctBnds indexing: Python 0-based.")
    else:
        raise ValueError(
            "slctBnds contains invalid indices for the 224-band endmember."
        )

    reference_spectrum = alunite_spectrum[selected_band_indices]

    if reference_spectrum.shape != (188,):
        raise ValueError(
            "The selected Alunite reference spectrum must have shape (188,); "
            f"received {reference_spectrum.shape}."
        )

    sam_angles = fast_sam_map(
        image_cube,
        reference_spectrum
    )

    expected_map_shape = (250, 190)

    if sam_angles.shape != expected_map_shape:
        raise ValueError(
            f"SAM map must have shape {expected_map_shape}; "
            f"received {sam_angles.shape}."
        )

    detection_map = detect_from_sam(
        sam_angles,
        SAM_THRESHOLD
    )

    binary_map = detection_map.astype(np.uint8)

    if binary_map.shape != expected_map_shape:
        raise ValueError(
            f"Binary map must have shape {expected_map_shape}; "
            f"received {binary_map.shape}."
        )

    unique_binary_values = set(np.unique(binary_map).tolist())

    if not unique_binary_values.issubset({0, 1}):
        raise ValueError(
            "Binary map values must contain only 0 and 1; "
            f"received {unique_binary_values}."
        )

    detected_pixels, total_pixels, detection_percentage = (
        calculate_detection_percentage(detection_map)
    )
    detected_pixels = int(detected_pixels)
    not_detected_pixels = total_pixels - detected_pixels

    if detected_pixels + not_detected_pixels != total_pixels:
        raise ValueError(
            "Detected and non-detected pixel counts do not equal total pixels."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_sam_map(
        sam_angles,
        os.path.join(OUTPUT_DIR, "alunite_sam_map.png"),
        title="Alunite SAM Angle Map"
    )

    save_detection_map(
        detection_map,
        os.path.join(OUTPUT_DIR, "alunite_detection_map.png"),
        title="Alunite Detection Map"
    )

    save_binary_map(
        binary_map,
        os.path.join(OUTPUT_DIR, "alunite_binary_map.png")
    )

    np.save(
        os.path.join(OUTPUT_DIR, "alunite_binary_matrix.npy"),
        binary_map
    )

    np.savetxt(
        os.path.join(OUTPUT_DIR, "alunite_binary_matrix.txt"),
        binary_map,
        fmt="%d"
    )

    background_image = np.mean(image_cube, axis=2)

    save_detection_overlay(
        background_image,
        binary_map,
        os.path.join(OUTPUT_DIR, "alunite_overlay.png")
    )

    wavelengths = np.asarray(
        ground_truth["waveLength"]
    ).reshape(-1)

    if wavelengths.size == endmembers.shape[0]:
        selected_wavelengths = wavelengths[selected_band_indices]
        x_axis = selected_wavelengths
        x_label = "Wavelength"
    else:
        x_axis = np.arange(1, 189)
        x_label = "Selected band index"

    plot_spectrum(
        x_axis,
        reference_spectrum,
        os.path.join(OUTPUT_DIR, "alunite_reference_spectrum.png"),
        title="Alunite Reference Spectrum"
    )

    detected_coordinates = np.argwhere(detection_map)

    if detected_coordinates.size > 0:
        pixel_row, pixel_col = detected_coordinates[0]
        detected_spectrum = image_cube[pixel_row, pixel_col, :]
        comparison_title = (
            f"Detected Pixel ({pixel_row}, {pixel_col}) vs Alunite"
        )
        plot_two_spectra(
            x_axis,
            detected_spectrum,
            reference_spectrum,
            os.path.join(OUTPUT_DIR, "alunite_pixel_comparison.png"),
            label1="Detected pixel",
            label2="Alunite reference"
        )
    else:
        plot_spectrum(
            x_axis,
            reference_spectrum,
            os.path.join(OUTPUT_DIR, "alunite_pixel_comparison.png"),
            title="No Detected Alunite Pixel"
        )
        comparison_title = "No detected pixel"

    results = (
        "Target mineral: Alunite\n"
        "Image dimensions: 250 x 190\n"
        "Number of spectral bands: 188\n"
        f"Total pixels: {total_pixels}\n"
        f"Detected Alunite pixels: {detected_pixels}\n"
        f"Not detected pixels: {not_detected_pixels}\n"
        f"Detection percentage: {detection_percentage:.6f}%\n"
        "SAM threshold: 0.15 radians\n"
        "Alunite endmember index: 1\n"
        f"Binary map shape: {binary_map.shape}\n"
        "Binary values: 0 = not detected, 1 = detected\n"
        f"Reference spectrum shape: {reference_spectrum.shape}\n"
        f"SAM map shape: {sam_angles.shape}\n"
    )

    with open(
        os.path.join(OUTPUT_DIR, "results.txt"),
        "w",
        encoding="utf-8"
    ) as results_file:
        results_file.write(results)

    print("\nEvaluation 1 results")
    print("Target mineral: Alunite")
    print("Image dimensions: 250 x 190")
    print("Number of spectral bands: 188")
    print("Total pixels:", total_pixels)
    print("Detected Alunite pixels:", detected_pixels)
    print("Not detected pixels:", not_detected_pixels)
    print(
        "Detection percentage:",
        f"{detection_percentage:.6f}% (spectral detection percentage)"
    )
    print("SAM threshold: 0.15 radians")
    print("Alunite endmember index: 1")
    print("Binary map shape:", binary_map.shape)
    print("Binary values: 0 = not detected, 1 = detected")
    print("Reference spectrum shape:", reference_spectrum.shape)
    print("SAM map shape:", sam_angles.shape)
    print("Comparison:", comparison_title)
    print("\n" + "=" * 60)
    print("ALUNITE BINARY SPATIAL DETECTION")
    print("=" * 60)
    print("Binary map shape:", binary_map.shape)
    print("Total pixels:", total_pixels)
    print("Alunite detected pixels:", detected_pixels)
    print("Alunite not detected pixels:", not_detected_pixels)
    print("Detection percentage:", f"{detection_percentage:.6f}%")
    print("Unique binary values:", sorted(unique_binary_values))
    print("=" * 60)
    print("\nOutputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":

    main()