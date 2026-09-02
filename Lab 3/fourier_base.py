import cv2
import numpy as np
from matplotlib import pyplot as plt


# -----------------------------
# Basic Fourier utilities
# -----------------------------
def fft_shifted(image):
    return np.fft.fftshift(np.fft.fft2(image))


def ifft_shifted(frequency_image):
    return np.real(np.fft.ifft2(np.fft.ifftshift(frequency_image)))


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def create_distance_matrix(shape):
    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.hypot(x - center_x, y - center_y)
    return distance


# -----------------------------
# Gaussian filters
# -----------------------------
def gaussian_low_pass_filter(image, d0):
    f = fft_shifted(image)
    d = create_distance_matrix(image.shape)
    h = np.exp(-(d ** 2) / (2 * (d0 ** 2)))
    filtered = f * h
    return ifft_shifted(filtered)


def gaussian_high_pass_filter(image, d0):
    f = fft_shifted(image)
    d = create_distance_matrix(image.shape)
    h = 1.0 - np.exp(-(d ** 2) / (2 * (d0 ** 2)))
    filtered = f * h
    return ifft_shifted(filtered)


def gaussian_band_reject_filter(image, d0, width):
    f = fft_shifted(image)
    d = create_distance_matrix(image.shape)
    h = 1.0 - np.exp(-((d ** 2 - d0 ** 2) / (d * width + 1e-6)) ** 2)
    filtered = f * h
    return ifft_shifted(filtered)


def gaussian_notch_reject_filter(image, notch_centers, sigma=10):
    f = fft_shifted(image)
    rows, cols = image.shape
    y, x = np.ogrid[:rows, :cols]
    h = np.ones((rows, cols), dtype=np.float32)

    for r0, c0 in notch_centers:
        dist2 = (y - r0) ** 2 + (x - c0) ** 2
        notch = np.exp(-(dist2) / (2 * (sigma ** 2)))
        h *= (1.0 - notch)

    filtered = f * h
    return ifft_shifted(filtered)


# -----------------------------
# Butterworth filters
# -----------------------------
def butterworth_low_pass_filter(image, d0, n=2):
    f = fft_shifted(image)
    d = create_distance_matrix(image.shape)
    h = 1.0 / (1.0 + (d / (d0 + 1e-6)) ** (2 * n))
    filtered = f * h
    return ifft_shifted(filtered)


def butterworth_high_pass_filter(image, d0, n=2):
    f = fft_shifted(image)
    d = create_distance_matrix(image.shape)
    h = 1.0 / (1.0 + (d0 / (d + 1e-6)) ** (2 * n))
    filtered = f * h
    return ifft_shifted(filtered)


def butterworth_band_reject_filter(image, d0, width, n=2):
    f = fft_shifted(image)
    d = create_distance_matrix(image.shape)
    numerator = (d * width) / (d ** 2 - d0 ** 2 + 1e-6)
    h = 1.0 / (1.0 + numerator ** (2 * n))
    filtered = f * h
    return ifft_shifted(filtered)


def butterworth_notch_reject_filter(image, notch_centers, d0, n=2):
    f = fft_shifted(image)
    rows, cols = image.shape
    y, x = np.ogrid[:rows, :cols]
    h = np.ones((rows, cols), dtype=np.float32)

    for r0, c0 in notch_centers:
        dist = np.hypot(x - c0, y - r0)
        notch = 1.0 / (1.0 + (d0 / (dist + 1e-6)) ** (2 * n))
        h *= notch

    filtered = f * h
    return ifft_shifted(filtered)


# -----------------------------
# Statistical filters
# -----------------------------
def contraharmonic_mean_filter(image, kernel_size, q):
    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(image, dtype=np.float32)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            window = padded[r:r + kernel_size, c:c + kernel_size].astype(np.float32)
            if q == 0:
                output[r, c] = np.mean(window)
            else:
                numerator = np.sum(np.power(window, q + 1))
                denominator = np.sum(np.power(window, q))
                if np.isclose(denominator, 0):
                    output[r, c] = window[kernel_size // 2, kernel_size // 2]
                else:
                    output[r, c] = numerator / denominator
    return output


def alpha_trimmed_mean_filter(image, kernel_size, alpha):
    if alpha <= 0 or alpha >= 0.5:
        raise ValueError("alpha must be between 0 and 0.5")

    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(image, dtype=np.float32)
    trim_count = int(alpha * kernel_size * kernel_size)
    trim_count = min(trim_count, kernel_size * kernel_size // 2)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            window = padded[r:r + kernel_size, c:c + kernel_size].astype(np.float32).ravel()
            sorted_window = np.sort(window)
            if trim_count == 0:
                output[r, c] = np.mean(sorted_window)
            else:
                trimmed = sorted_window[trim_count:-trim_count]
                output[r, c] = np.mean(trimmed)

    return output


# -----------------------------
# Higher-level demo
# -----------------------------
def show_results(results):
    cols = 3
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for ax, (title, image) in zip(axes, results.items()):
        ax.imshow(normalize_image(image), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def demo_all_filters(image_path='pnois2.jpg'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = {
        'Original': image,
        'Gaussian LPF': gaussian_low_pass_filter(image, d0=30),
        'Gaussian HPF': gaussian_high_pass_filter(image, d0=30),
        'Gaussian BRF': gaussian_band_reject_filter(image, d0=100, width=20),
        'Gaussian Notch Reject': gaussian_notch_reject_filter(image, notch_centers=[(120, 150), (180, 220)], sigma=15),
        'Butterworth LPF': butterworth_low_pass_filter(image, d0=30, n=2),
        'Butterworth HPF': butterworth_high_pass_filter(image, d0=30, n=2),
        'Butterworth BRF': butterworth_band_reject_filter(image, d0=100, width=20, n=2),
        'Butterworth Notch Reject': butterworth_notch_reject_filter(image, notch_centers=[(120, 150), (180, 220)], d0=20, n=2),
        'CMF (Q=1.5)': contraharmonic_mean_filter(image, kernel_size=3, q=1.5),
        'Alpha-trimmed mean (alpha=0.1)': alpha_trimmed_mean_filter(image, kernel_size=5, alpha=0.1),
    }

    show_results(results)


if __name__ == "__main__":
    demo_all_filters()
