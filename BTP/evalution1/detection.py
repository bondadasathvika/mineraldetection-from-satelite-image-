import numpy as np


def detect_from_sam(
    sam_angles,
    threshold
):

    detection_map = (
        sam_angles <= threshold
    )

    return detection_map


def calculate_detection_percentage(
    detection_map
):

    total_pixels = detection_map.size

    detected_pixels = np.sum(
        detection_map
    )

    percentage = (
        detected_pixels
        /
        total_pixels
    ) * 100

    return (
        detected_pixels,
        total_pixels,
        percentage
    )


def get_detected_coordinates(
    detection_map
):

    return np.argwhere(
        detection_map
    )