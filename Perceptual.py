import numpy as np

def compute_masking_threshold(spectrum, sample_rate, threshold_percentage=90):
    freqs = np.linspace(0, sample_rate / 2, len(spectrum))

    # A simplified example: masking threshold decreases with frequency (logarithmic model)
    masking_threshold = 1 / (freqs + 1)

    # Normalize threshold to fit the spectrum magnitude range
    masking_threshold *= np.max(np.abs(spectrum)) / np.max(masking_threshold)

    # Adjust threshold by the specified percentage
    masking_threshold *= threshold_percentage / 100.0

    return masking_threshold


def apply_perceptual_masking(spectrum, masking_threshold):
    masked_spectrum = np.where(np.abs(spectrum) >= masking_threshold, spectrum, 0)
    return masked_spectrum

def compute_subband_masking_threshold(subbands):
    num_subbands = subbands.shape[0]

    # Simplified model: stronger masking for lower subbands
    masking_threshold = 1 / (np.arange(1, num_subbands + 1))

    # Normalize thresholds
    masking_threshold *= np.max(np.abs(subbands)) / np.max(masking_threshold)

    return masking_threshold