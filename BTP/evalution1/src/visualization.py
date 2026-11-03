import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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


def save_binary_map(
    binary_map,
    output_path,
    title="Alunite Binary Detection Map"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    colors = ["black", "yellow"]
    legend_handles = [
        Patch(facecolor=colors[1], edgecolor="black", label="1 = Alunite detected"),
        Patch(facecolor=colors[0], edgecolor="white", label="0 = Alunite not detected")
    ]

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        binary_map,
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_detection_overlay(
    background_image,
    binary_map,
    output_path,
    title="Alunite Detection Overlay"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    detected_overlay = np.ma.masked_where(
        binary_map == 0,
        binary_map
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        background_image,
        cmap="gray"
    )
    plt.imshow(
        detected_overlay,
        cmap=ListedColormap(["red"]),
        alpha=0.75,
        interpolation="nearest"
    )

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend(
        handles=[Patch(facecolor="red", edgecolor="red", label="Alunite detected")],
        loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()