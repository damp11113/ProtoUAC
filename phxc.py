# PHXC is Parametric Harmonic eXtraction Coding

import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import get_window, find_peaks
import struct
import math
from typing import List, Dict

class HarmonicExtractor:
    """
    Extracts multiple primary frequencies (P0) and their harmonic components
    from a single audio chunk, supporting polyphonic analysis.

    It returns a list of 'Harmonic' objects (the Chunk), where each object
    represents one detected primary frequency source.

    Optimized version with optional logarithmic frequency scanning.
    """

    def __init__(self,
                 sample_rate: int,
                 window_size: int,
                 hop_size: int,
                 min_f0_freq: float,
                 max_f0_freq: float,
                 peak_threshold: float,
                 max_harmonics_per_f0: int,
                 max_harmonic_freq_output: float,
                 max_harmonic_freq_object: int = 10,
                 log: bool = False):
        """
        Initializes the Harmonic Extractor with analysis parameters.

        Parameters:
        - window_size: Increased for better frequency resolution, essential for high F0.
        - max_harmonic_freq_object: Limits the number of Harmonic Objects (P0 sources)
          returned per chunk by prioritizing the strongest ones.
        - log: If True, uses logarithmic frequency scanning for better pitch detection
          across wide frequency ranges.
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.min_f0_freq = min_f0_freq
        self.max_f0_freq = max_f0_freq
        self.peak_threshold = peak_threshold
        self.max_harmonics_per_f0 = max_harmonics_per_f0
        self.max_harmonic_freq_output = max_harmonic_freq_output
        self.max_harmonic_freq_object = max_harmonic_freq_object
        self.log = log

        # Pre-compute window to avoid recalculation
        self.window = get_window('hann', window_size)
        self.fft_size = window_size

        # Frequency resolution (Hz per bin) - Higher window_size improves this.
        self.freq_resolution = sample_rate / self.fft_size

        # Calculate FFT bin indices for the P0 search range
        self.min_f0_bin = int(self.min_f0_freq / self.freq_resolution)
        self.max_f0_bin = min(int(self.max_f0_freq / self.freq_resolution), self.fft_size // 2)

        # Pre-compute log frequency bins if log mode is enabled
        if self.log:
            self._precompute_log_bins()

        # Pre-compute frequency array for faster lookups
        self._freq_array = np.arange(self.fft_size // 2) * self.freq_resolution

    def _precompute_log_bins(self):
        """Pre-compute logarithmically spaced frequency bins for log mode."""
        # Number of bins per octave (adjustable for resolution)
        bins_per_octave = 12  # Similar to musical semitones

        # Calculate number of octaves in the range
        octaves = np.log2(self.max_f0_freq / self.min_f0_freq)
        num_log_bins = int(octaves * bins_per_octave)

        # Create logarithmically spaced frequencies
        self.log_freqs = np.logspace(
            np.log10(self.min_f0_freq),
            np.log10(self.max_f0_freq),
            num_log_bins
        )

        # Convert to FFT bin indices
        self.log_bins = np.round(self.log_freqs / self.freq_resolution).astype(int)
        # Remove duplicates and clip to valid range
        self.log_bins = np.unique(np.clip(self.log_bins, 0, self.fft_size // 2 - 1))

    def process_chunk(self, audio_chunk: np.ndarray) -> List[Dict]:
        """
        Processes a single audio chunk and returns a list of Harmonic Objects (the Chunk).

        Optimizations:
        - Vectorized operations where possible
        - Reduced redundant calculations
        - Optional log-scale frequency scanning
        """
        # Fast padding using numpy
        if len(audio_chunk) != self.window_size:
            padded_chunk = np.zeros(self.window_size, dtype=audio_chunk.dtype)
            padded_chunk[:len(audio_chunk)] = audio_chunk
        else:
            padded_chunk = audio_chunk

        # 1. Apply Windowing and FFT (in-place multiplication for speed)
        windowed_data = padded_chunk * self.window
        fft_result = np.fft.rfft(windowed_data, n=self.fft_size)  # Use rfft for real signals

        # Get magnitude and phase (rfft already gives us only positive frequencies)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)

        # Normalize magnitude to find peaks relative to the global max
        max_mag = fft_magnitude.max()

        if max_mag == 0:  # Handle silence
            return []

        # --- Multi-Pitch Detection (Peak Search) ---

        if self.log:
            # Log-scale frequency scanning
            primary_freq_bins, primary_magnitudes = self._find_peaks_log_scale(
                fft_magnitude, max_mag
            )
        else:
            # Linear frequency scanning
            primary_freq_bins, primary_magnitudes = self._find_peaks_linear(
                fft_magnitude, max_mag
            )

        if len(primary_freq_bins) == 0:
            return []

        # --- Limit to top N strongest peaks ---
        if len(primary_freq_bins) > self.max_harmonic_freq_object:
            # Use argpartition for faster partial sorting
            top_indices = np.argpartition(primary_magnitudes, -self.max_harmonic_freq_object)[
                          -self.max_harmonic_freq_object:]
            # Sort only the top N
            top_indices = top_indices[np.argsort(primary_magnitudes[top_indices])[::-1]]
            primary_freq_bins = primary_freq_bins[top_indices]
            primary_magnitudes = primary_magnitudes[top_indices]

        # --- Harmonic Extraction for Each Primary Peak ---
        harmonic_objects = self._extract_harmonics(
            primary_freq_bins, primary_magnitudes, fft_magnitude, fft_phase, max_mag
        )

        return harmonic_objects

    def _find_peaks_linear(self, fft_magnitude: np.ndarray, max_mag: float):
        """Find peaks using linear frequency scanning."""
        search_start = max(0, self.min_f0_bin)
        search_end = min(len(fft_magnitude), self.max_f0_bin)

        if search_start >= search_end:
            return np.array([]), np.array([])

        # Find peaks that exceed the absolute threshold
        peaks, _ = find_peaks(
            fft_magnitude[search_start:search_end],
            height=self.peak_threshold * max_mag
        )

        # Convert relative peak indices back to absolute bin indices
        primary_freq_bins = peaks + search_start
        primary_magnitudes = fft_magnitude[primary_freq_bins]

        return primary_freq_bins, primary_magnitudes

    def _find_peaks_log_scale(self, fft_magnitude: np.ndarray, max_mag: float):
        """Find peaks using logarithmic frequency scanning."""
        # Sample magnitudes at log-spaced bins
        log_magnitudes = fft_magnitude[self.log_bins]

        # Find peaks in the log-spaced samples
        peaks, _ = find_peaks(
            log_magnitudes,
            height=self.peak_threshold * max_mag
        )

        if len(peaks) == 0:
            return np.array([]), np.array([])

        # Get the actual FFT bins corresponding to these peaks
        primary_freq_bins = self.log_bins[peaks]

        # Refine peak locations using parabolic interpolation on nearby bins
        refined_bins = []
        refined_mags = []

        for bin_idx in primary_freq_bins:
            if bin_idx > 0 and bin_idx < len(fft_magnitude) - 1:
                # Parabolic interpolation for sub-bin accuracy
                alpha = fft_magnitude[bin_idx - 1]
                beta = fft_magnitude[bin_idx]
                gamma = fft_magnitude[bin_idx + 1]

                # Check if peak is valid
                if beta > alpha and beta > gamma:
                    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                    refined_bin = bin_idx + p
                    # Interpolated magnitude
                    refined_mag = beta - 0.25 * (alpha - gamma) * p

                    refined_bins.append(int(np.round(refined_bin)))
                    refined_mags.append(refined_mag)
            else:
                refined_bins.append(bin_idx)
                refined_mags.append(fft_magnitude[bin_idx])

        return np.array(refined_bins), np.array(refined_mags)

    def _extract_harmonics(self, primary_freq_bins: np.ndarray,
                           primary_magnitudes: np.ndarray,
                           fft_magnitude: np.ndarray,
                           fft_phase: np.ndarray,
                           max_mag: float) -> List[Dict]:
        """Extract harmonics for all detected primary frequencies."""
        harmonic_objects = []
        threshold = self.peak_threshold * max_mag
        half_fft_size = len(fft_magnitude)
        nyquist_freq = self.sample_rate / 2

        for p0_bin, p0_amp in zip(primary_freq_bins, primary_magnitudes):
            # Skip if the primary frequency is too weak
            if p0_amp < threshold:
                continue

            p0_freq = p0_bin * self.freq_resolution
            p0_phase = fft_phase[p0_bin]

            # Pre-allocate list for harmonics
            current_harmonics = [{'amp': float(p0_amp), 'phase': float(p0_phase)}]

            # Vectorized harmonic frequency calculation
            harmonic_numbers = np.arange(2, self.max_harmonics_per_f0 + 1)
            target_freqs = harmonic_numbers * p0_freq

            # Filter harmonics by frequency limits
            valid_mask = (target_freqs <= self.max_harmonic_freq_output) & (target_freqs < nyquist_freq)
            valid_harmonics = harmonic_numbers[valid_mask]

            if len(valid_harmonics) == 0:
                # Only fundamental frequency
                harmonic_objects.append({
                    "freq": float(p0_freq),
                    "n_harmonic": 1,
                    "main_amp": float(p0_amp),
                    "harmonics": current_harmonics
                })
                continue

            # Calculate bins for all valid harmonics at once
            harmonic_bins = (valid_harmonics * p0_freq / self.freq_resolution).astype(int)
            harmonic_bins = harmonic_bins[harmonic_bins < half_fft_size]

            # Extract amplitudes and phases for all harmonics at once
            if len(harmonic_bins) > 0:
                harmonic_amps = fft_magnitude[harmonic_bins]
                harmonic_phases = fft_phase[harmonic_bins]

                # Add harmonics to list
                for amp, phase in zip(harmonic_amps, harmonic_phases):
                    current_harmonics.append({'amp': float(amp), 'phase': float(phase)})

            # Create the Harmonic Object
            harmonic_objects.append({
                "freq": float(p0_freq),
                "n_harmonic": len(current_harmonics),
                "main_amp": float(p0_amp),
                "harmonics": current_harmonics
            })

        return harmonic_objects


# --- 2. HARMONIC GENERATOR (SYNTHESIS) ---

class HarmonicGenerator:
    """
    FAST version: fully vectorized harmonic synthesis.
    """
    def __init__(self, sample_rate: int, window_size: int, hop_size: int):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size

        self.window = get_window('hann', window_size)

        # Precompute time base ONCE
        self.t = np.arange(window_size, dtype=np.float32) / sample_rate

    def process_chunk(self, harmonic_objects_chunk: list) -> np.ndarray:
        if not harmonic_objects_chunk:
            return np.zeros(self.window_size, dtype=np.float32)

        freqs = []
        amps = []
        phases = []

        for h_obj in harmonic_objects_chunk:
            p0 = h_obj['freq']
            for n, h in enumerate(h_obj['harmonics'], start=1):
                freqs.append(p0 * n)
                amps.append(h['amp'])
                phases.append(h['phase'])

        freqs = np.array(freqs, dtype=np.float32)
        amps = np.array(amps, dtype=np.float32)
        phases = np.array(phases, dtype=np.float32)

        sine_matrix = amps[:, None] * np.sin(
            2 * np.pi * freqs[:, None] * self.t[None, :] + phases[:, None]
        )

        # Sum over all harmonics
        output_chunk = sine_matrix.sum(axis=0).astype(np.float32)

        return output_chunk #* self.window