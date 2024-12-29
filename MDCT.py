import numpy as np
from scipy.signal.windows import kaiser

def mdct4_mono(x):  # alpha is the window shape parameter for Kaiser-Bessel Derived window
    """
        Perform MDCT (Modified Discrete Cosine Transform) on mono audio signal.

        Parameters:
            x (np.ndarray): Mono audio signal as a 1D NumPy array. The length must be a multiple of 4.

        Returns:
            np.ndarray: MDCT coefficients.
        """
    if len(x) % 4 != 0:
        raise ValueError("MDCT requires the input length to be a multiple of 4.")

    N = len(x)
    M = N // 2
    N4 = N // 4

    # Apply pre-rotation
    rot = np.roll(x, N4)
    rot[:N4] = -rot[:N4]

    # Compute MDCT
    t = np.arange(0, N4)
    w = np.exp(-1j * 2 * np.pi * (t + 1. / 8.) / N)
    c = np.take(rot, 2 * t) - np.take(rot, N - 2 * t - 1) \
        - 1j * (np.take(rot, M + 2 * t) - np.take(rot, M - 2 * t - 1))
    c = (2. / np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)

    # Pack coefficients
    y = np.zeros(M)
    y[2 * t] = np.real(c[t])
    y[M - 2 * t - 1] = -np.imag(c[t])

    return y


def imdct4_mono(x):
    """
        Perform IMDCT (Inverse Modified Discrete Cosine Transform) on MDCT coefficients.

        Parameters:
            x (np.ndarray): MDCT coefficients as a 1D NumPy array. The length must be even.

        Returns:
            np.ndarray: Reconstructed mono audio signal.
        """
    if len(x) % 2 != 0:
        raise ValueError("IMDCT requires the input length to be even.")

    N = len(x)
    M = N // 2
    N2 = N * 2

    # Compute IMDCT
    t = np.arange(0, M)
    w = np.exp(-1j * 2 * np.pi * (t + 1. / 8.) / N2)
    c = np.take(x, 2 * t) + 1j * np.take(x, N - 2 * t - 1)
    c = 0.5 * w * c
    c = np.fft.fft(c, M)
    c = ((8 / np.sqrt(N2)) * w) * c

    # Prepare rotated output
    rot = np.zeros(N2)
    rot[2 * t] = np.real(c[t])
    rot[N + 2 * t] = np.imag(c[t])

    t = np.arange(1, N2, 2)
    rot[t] = -rot[N2 - t - 1]

    # Post-rotation
    y = np.zeros(N2)
    t = np.arange(0, 3 * M)
    y[t] = rot[t + M]
    t = np.arange(3 * M, N2)
    y[t] = -rot[t - 3 * M]

    return y

def mdct4(x):
    N = x.shape[0]
    print(N)
    if N % 4 != 0:
        raise ValueError("MDCT4 only defined for vectors of length multiple of four.")
    M = N // 2
    N4 = N // 4

    rot = np.roll(x, N4)
    rot[:N4] = -rot[:N4]
    t = np.arange(0, N4)
    w = np.exp(-1j * 2 * np.pi * (t + 1. / 8.) / N)
    c = np.take(rot, 2 * t) - np.take(rot, N - 2 * t - 1) - 1j * (np.take(rot, M + 2 * t) - np.take(rot, M - 2 * t - 1))
    c = (2. / np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)
    y = np.zeros(M)
    y[2 * t] = np.real(c[t])
    y[M - 2 * t - 1] = -np.imag(c[t])
    return y

def imdct4(x):
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError("iMDCT4 only defined for even-length vectors.")
    M = N // 2
    N2 = N * 2

    t = np.arange(0, M)
    w = np.exp(-1j * 2 * np.pi * (t + 1. / 8.) / N2)
    c = np.take(x, 2 * t) + 1j * np.take(x, N - 2 * t - 1)
    c = 0.5 * w * c
    c = np.fft.fft(c, M)
    c = ((8 / np.sqrt(N2)) * w) * c

    rot = np.zeros(N2)

    rot[2 * t] = np.real(c[t])
    rot[N + 2 * t] = np.imag(c[t])

    t = np.arange(1, N2, 2)
    rot[t] = -rot[N2 - t - 1]

    t = np.arange(0, 3 * M)
    y = np.zeros(N2)
    y[t] = rot[t + M]
    t = np.arange(3 * M, N2)
    y[t] = -rot[t - 3 * M]
    return y


def mdct(x):
    N = len(x)
    M = N // 2
    # Perform FFT, skipping the cosine window step
    return np.fft.rfft(x)[1:M+1]  # Return the positive frequencies

def imdct(X):
    M = len(X)
    N = M * 2
    # Reconstruct the signal by applying inverse FFT without the window
    x_reconstructed = np.fft.irfft(np.hstack(([0], X, [0])))  # Apply inverse FFT with zero padding
    return x_reconstructed[:N]  # Return the full signal without windowing