from typing import Tuple
import numpy as np
from scipy import signal


def detect_max_freq_response(fft_result, freq, noise_threshold_db):
    """
    Detect the maximum frequency with significant energy above the noise threshold.
    Uses a more sophisticated method to find actual signal rolloff.

    Args:
        fft_result: FFT result (complex array)
        freq: Frequency bins corresponding to FFT result

    Returns:
        Maximum frequency in Hz with significant energy
    """
    # Convert to magnitude (linear scale)
    magnitude = np.abs(fft_result)

    # Convert to dB scale
    # Add small epsilon to avoid log(0)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Find the peak magnitude (reference level)
    peak_db = np.max(magnitude_db)

    # Calculate relative threshold (dB below peak)
    # Looking for frequencies that are within a certain range of the peak
    relative_threshold = peak_db + noise_threshold_db

    # Find frequencies above the relative threshold
    above_threshold = magnitude_db > relative_threshold

    if np.any(above_threshold):
        # Find the highest frequency above threshold
        max_freq_idx = np.where(above_threshold)[0][-1]
        max_freq = freq[max_freq_idx]

        return int(max_freq)
    else:
        return 0

def design_lowpass_filter(cutoff_freq: float, sample_rate: int, order: int = 8) -> Tuple:
    """Design a Butterworth lowpass filter."""
    nyquist = sample_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass(audio: np.ndarray, b, a) -> np.ndarray:
    """Apply lowpass filter to audio using pre-designed filter coefficients."""
    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio)
    else:
        filtered = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            filtered[:, ch] = signal.filtfilt(b, a, audio[:, ch])
        return filtered
