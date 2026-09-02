"""
CSE 4102 - Computer Graphics and Image Processing
Lab 02 - A2

Histogram Matching

Task:
Given a source/input RGB image and a reference RGB image,
perform histogram matching by operating ONLY on the L channel
in Lab color space.

This script:
- loads the source and reference images,
- converts both to Lab,
- computes histogram, PDF, and CDF for the L channel,
- builds a manual mapping from source intensities to reference intensities,
- applies the mapping only to the source L channel,
- keeps the source a and b channels unchanged,
- saves the output image, and
- shows the result in a 3 x 3 layout like the sample figure.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / "input.jpg"
REFERENCE_PATH = SCRIPT_DIR / "reference.jpg"
OUTPUT_PATH = SCRIPT_DIR / "A2_output.jpg"


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and return both BGR and RGB versions."""

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image:\n{path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


def calculate_histogram_pdf_cdf(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate histogram, PDF, and CDF for one 8-bit channel."""

    pixels = channel.ravel()

    histogram = np.bincount(pixels, minlength=256)

    pdf = histogram / pixels.size

    cdf = np.cumsum(pdf)

    return histogram, pdf, cdf


def create_histogram_matching_mapping(source_cdf: np.ndarray, reference_cdf: np.ndarray) -> np.ndarray:
    """Create a mapping from source intensity values to reference intensity values.

    For each source intensity r, we find the reference intensity z whose
    CDF value is closest to source_cdf[r]. This is the manual histogram
    matching transformation used in the lab.
    """

    mapping = np.zeros(256, dtype=np.uint8)

    for r in range(256):
        difference = np.abs(reference_cdf - source_cdf[r])
        z = np.argmin(difference)
        mapping[r] = z

    return mapping


def print_results_info(source_l: np.ndarray, reference_l: np.ndarray, output_l: np.ndarray) -> None:
    """Print useful information for the lab viva."""

    print("\n========== A2 RESULTS ==========")
    print(f"Source L range:     {source_l.min()} - {source_l.max()}")
    print(f"Reference L range:  {reference_l.min()} - {reference_l.max()}")
    print(f"Output L range:     {output_l.min()} - {output_l.max()}")
    print("================================")


def plot_results(
    source_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    output_rgb: np.ndarray,
    source_pdf: np.ndarray,
    source_cdf: np.ndarray,
    reference_pdf: np.ndarray,
    reference_cdf: np.ndarray,
    output_pdf: np.ndarray,
    output_cdf: np.ndarray,
) -> None:
    """Display the source, reference, and output images with their graphs."""

    levels = np.arange(256)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("A2 - Histogram Matching Using L Channel in Lab Color Space", fontsize=18)

    # Row 1: Source image and its graphs
    axes[0, 0].imshow(source_rgb)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].plot(levels, source_pdf, color="red")
    axes[0, 1].set_title("Source PDF")
    axes[0, 1].set_xlabel("Intensity")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].set_xlim(0, 255)

    axes[0, 2].plot(levels, source_cdf, color="black")
    axes[0, 2].set_title("Source CDF - S(r)")
    axes[0, 2].set_xlabel("Intensity")
    axes[0, 2].set_ylabel("CDF")
    axes[0, 2].set_xlim(0, 255)
    axes[0, 2].set_ylim(0, 1.05)

    # Row 2: Reference image and its graphs
    axes[1, 0].imshow(reference_rgb)
    axes[1, 0].set_title("Reference Image")
    axes[1, 0].axis("off")

    axes[1, 1].plot(levels, reference_pdf, color="green")
    axes[1, 1].set_title("Reference PDF")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_xlim(0, 255)

    axes[1, 2].plot(levels, reference_cdf, color="green")
    axes[1, 2].set_title("Reference CDF - G(z)")
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("CDF")
    axes[1, 2].set_xlim(0, 255)
    axes[1, 2].set_ylim(0, 1.05)

    # Row 3: Output image and its graphs
    axes[2, 0].imshow(output_rgb)
    axes[2, 0].set_title("Output Image")
    axes[2, 0].axis("off")

    axes[2, 1].plot(levels, output_pdf, color="blue")
    axes[2, 1].set_title("Output PDF")
    axes[2, 1].set_xlabel("Intensity")
    axes[2, 1].set_ylabel("Probability")
    axes[2, 1].set_xlim(0, 255)

    axes[2, 2].plot(levels, output_cdf, color="blue")
    axes[2, 2].set_title("Output CDF")
    axes[2, 2].set_xlabel("Intensity")
    axes[2, 2].set_ylabel("CDF")
    axes[2, 2].set_xlim(0, 255)
    axes[2, 2].set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the figure only when using an interactive backend.
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """Run the full histogram matching pipeline."""

    print("=" * 60)
    print("CSE 4102 - A2 HISTOGRAM MATCHING")
    print("=" * 60)

    # Load images.
    source_bgr, source_rgb = load_image(SOURCE_PATH)
    reference_bgr, reference_rgb = load_image(REFERENCE_PATH)

    print("\nSource image:")
    print(SOURCE_PATH)
    print("Size:", source_bgr.shape)

    print("\nReference image:")
    print(REFERENCE_PATH)
    print("Size:", reference_bgr.shape)

    # Convert both images to Lab.
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2Lab)
    reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2Lab)

    # Split the Lab channels.
    source_L, source_a, source_b = cv2.split(source_lab)
    reference_L, _, _ = cv2.split(reference_lab)

    # Compute histogram statistics for source and reference L channels.
    source_hist, source_pdf, source_cdf = calculate_histogram_pdf_cdf(source_L)
    reference_hist, reference_pdf, reference_cdf = calculate_histogram_pdf_cdf(reference_L)

    # Build the mapping and apply it to the source L channel.
    mapping = create_histogram_matching_mapping(source_cdf, reference_cdf)
    output_L = mapping[source_L]

    # Compute output statistics.
    output_hist, output_pdf, output_cdf = calculate_histogram_pdf_cdf(output_L)

    # Merge the new L channel with the original source a and b channels.
    output_lab = cv2.merge((output_L, source_a, source_b))

    # Convert back to BGR for saving and RGB for plotting.
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    # Save output image.
    cv2.imwrite(str(OUTPUT_PATH), output_bgr)
    print("\nOutput image saved:")
    print(OUTPUT_PATH)

    # Print results.
    print_results_info(source_L, reference_L, output_L)

    # Plot and save the graphs.
    plot_results(
        source_rgb=source_rgb,
        reference_rgb=reference_rgb,
        output_rgb=output_rgb,
        source_pdf=source_pdf,
        source_cdf=source_cdf,
        reference_pdf=reference_pdf,
        reference_cdf=reference_cdf,
        output_pdf=output_pdf,
        output_cdf=output_cdf,
    )


if __name__ == "__main__":
    main()