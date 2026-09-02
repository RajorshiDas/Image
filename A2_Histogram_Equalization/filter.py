
# Fourier Transform - Frequency Domain Filters

import cv2
import numpy as np
from matplotlib import pyplot as plt


# ============================================================
# 1. TAKE INPUT
# ============================================================

img_input = cv2.imread('pnois2.jpg', 0)
img = img_input.copy()


# ============================================================
# 2. FOURIER TRANSFORM
# ============================================================

ft = np.fft.fft2(img)

# Move zero frequency to center
ft_shift = np.fft.fftshift(ft)


# ============================================================
# 3. MAGNITUDE AND PHASE
# ============================================================

magnitude_spectrum_ac = np.abs(ft_shift)

ang = np.angle(ft_shift)


# For visualization only
magnitude_spectrum = 20 * np.log(
    np.abs(ft_shift) + 1
)

magnitude_spectrum = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)

# Phase visualization
ang_ = cv2.normalize(
    ang,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)


# ============================================================
# 4. CREATE FREQUENCY COORDINATES
# ============================================================

rows, cols = img.shape

crow = rows // 2
ccol = cols // 2

u = np.arange(rows) - crow
v = np.arange(cols) - ccol

V, U = np.meshgrid(v, u)

# Distance from center
D = np.sqrt(U**2 + V**2)


# ============================================================
# 5. SELECT FILTER
# ============================================================

filter_type = "GLPF"

# Options:
#
# "GLPF"
# "GHPF"
# "GBRF"
# "GNRF"
# "BLPF"
# "BHPF"
# "BBRF"
# "BNRF"


# ============================================================
# 6. APPLY FILTER
# ============================================================


# ------------------------------------------------------------
# Gaussian Low-Pass Filter
# ------------------------------------------------------------

if filter_type == "GLPF":

    D0 = 50

    H = np.exp(
        -(D**2) / (2 * D0**2)
    )

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Gaussian High-Pass Filter
# ------------------------------------------------------------

elif filter_type == "GHPF":

    D0 = 50

    H_low = np.exp(
        -(D**2) / (2 * D0**2)
    )

    H = 1 - H_low

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Gaussian Band-Reject Filter
# ------------------------------------------------------------

elif filter_type == "GBRF":

    D0 = 50
    W = 20

    H = 1 - np.exp(
        -((D**2 - D0**2) /
          (D * W + 1e-10))
    )

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Gaussian Notch-Reject Filter
# ------------------------------------------------------------

elif filter_type == "GNRF":

    u0 = 50
    v0 = 50

    D0 = 10

    D1 = np.sqrt(
        (U - u0)**2 +
        (V - v0)**2
    )

    D2 = np.sqrt(
        (U + u0)**2 +
        (V + v0)**2
    )

    H1 = 1 - np.exp(
        -(D1**2) / (2 * D0**2)
    )

    H2 = 1 - np.exp(
        -(D2**2) / (2 * D0**2)
    )

    H = H1 * H2

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Butterworth Low-Pass Filter
# ------------------------------------------------------------

elif filter_type == "BLPF":

    D0 = 50
    n = 2

    H = 1 / (
        1 + (D / D0)**(2*n)
    )

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Butterworth High-Pass Filter
# ------------------------------------------------------------

elif filter_type == "BHPF":

    D0 = 50
    n = 2

    H_low = 1 / (
        1 + (D / D0)**(2*n)
    )

    H = 1 - H_low

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Butterworth Band-Reject Filter
# ------------------------------------------------------------

elif filter_type == "BBRF":

    D0 = 50
    W = 20
    n = 2

    denominator = D**2 - D0**2

    H = 1 / (
        1 +
        ((D * W) /
         (denominator + 1e-10))**(2*n)
    )

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ------------------------------------------------------------
# Butterworth Notch-Reject Filter
# ------------------------------------------------------------

elif filter_type == "BNRF":

    u0 = 50
    v0 = 50

    D0 = 10
    n = 2

    D1 = np.sqrt(
        (U - u0)**2 +
        (V - v0)**2
    )

    D2 = np.sqrt(
        (U + u0)**2 +
        (V + v0)**2
    )

    H1 = 1 / (
        1 +
        (D0 / (D1 + 1e-10))**(2*n)
    )

    H2 = 1 / (
        1 +
        (D0 / (D2 + 1e-10))**(2*n)
    )

    H = H1 * H2

    magnitude_spectrum_ac = (
        magnitude_spectrum_ac * H
    )


# ============================================================
# 7. COMBINE MAGNITUDE + PHASE
# ============================================================

final_result = np.multiply(
    magnitude_spectrum_ac,
    np.exp(1j * ang)
)


# ============================================================
# 8. INVERSE FOURIER TRANSFORM
# ============================================================

img_back = np.real(
    np.fft.ifft2(
        np.fft.ifftshift(final_result)
    )
)


# Normalize result for display
img_back_scaled = cv2.normalize(
    img_back,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)


# ============================================================
# 9. DISPLAY
# ============================================================

cv2.imshow("Input", img_input)

cv2.imshow(
    "Magnitude Spectrum",
    magnitude_spectrum
)

cv2.imshow(
    "Phase",
    ang_
)

cv2.imshow(
    "Inverse Transform - " + filter_type,
    img_back_scaled
)

cv2.waitKey(0)

cv2.destroyAllWindows()
