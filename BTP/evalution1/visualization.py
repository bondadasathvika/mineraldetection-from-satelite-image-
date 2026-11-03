import os
import numpy as np
import matplotlib.pyplot as plt


def save_sam_map(
    sam_map,
    output_path,
    title="Spectral Angle Map"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        sam_map
    )

    plt.title(title)

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.colorbar(
        label="SAM angle (radians)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def save_detection_map(
    detection_map,
    output_path,
    title="Mineral Detection Map"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        detection_map
    )

    plt.title(title)

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.colorbar(
        label="Detection"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def plot_spectrum(
    wavelengths,
    spectrum,
    output_path,
    title="Spectrum"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        wavelengths,
        spectrum
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Reflectance"
    )

    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def plot_two_spectra(
    wavelengths,
    spectrum1,
    spectrum2,
    output_path,
    label1="Image Pixel",
    label2="Reference Mineral"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        wavelengths,
        spectrum1,
        label=label1
    )

    plt.plot(
        wavelengths,
        spectrum2,
        label=label2
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Reflectance"
    )

    plt.title(
        "Spectral Signature Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()