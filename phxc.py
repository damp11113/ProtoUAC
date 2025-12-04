# PHXC is Parametric Harmonic eXtraction Coding

import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import get_window, find_peaks
import struct
import math
from typing import List, Dict


# --- 1. HARMONIC EXTRACTOR (ANALYSIS) ---

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
                 log: bool = True):
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


# --- 3. CHUNK-BY-CHUNK PROCESSOR AND DEMO ---

def process_chunk_by_chunk(input_audio_path: str, output_audio_path: str, params: dict):
    """
    Reads a WAV file chunk by chunk, extracts Chunks (list of Harmonic Objects),
    then generates the output audio chunk by chunk using the Overlap-Add method.
    """
    from packer import HarmonicPacker

    print(f"--- Starting Analysis of {input_audio_path} ---")

    try:
        sample_rate, input_data = read(input_audio_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_audio_path}")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Data type conversion and normalization
    if input_data.ndim > 1:
        input_data = input_data.mean(axis=1)
    if input_data.dtype.kind == 'i':
        max_val = np.iinfo(input_data.dtype).max
        audio_float = input_data.astype(np.float32) / max_val
    else:
        audio_float = input_data.astype(np.float32)

    if audio_float.size == 0:
        print("Input audio data is empty.")
        return

    # --- Setup Parameters ---
    WINDOW_SIZE = params['window_size']
    HOP_SIZE = params['hop_size']

    # Initialize Extractor and Generator
    extractor = HarmonicExtractor(
        sample_rate=sample_rate,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        min_f0_freq=params['min_f0_freq'],
        max_f0_freq=params['max_f0_freq'],
        peak_threshold=params['peak_threshold'],
        max_harmonics_per_f0=params['max_harmonics_per_f0'],
        max_harmonic_freq_output=params['max_harmonic_freq_output'],
        max_harmonic_freq_object=params['max_harmonic_freq_object']
    )
    generator = HarmonicGenerator(
        sample_rate=sample_rate,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE
    )

    extracted_parameters = []

    output_signal_length = audio_float.size + WINDOW_SIZE
    output_signal = np.zeros(output_signal_length, dtype=np.float32)

    packer_compact = HarmonicPacker(
        sample_rate=sample_rate,
        window_size=WINDOW_SIZE,
        amp_dtype='uint8',
        phase_dtype='uint8',
        scale_mode='log'
    )

    # --- Analysis Loop (Extraction) ---
    print("Extracting Chunks (list of Harmonic Objects) chunk by chunk...")
    maxvalue = 0

    for i in range(0, audio_float.size, HOP_SIZE):
        chunk_start = i
        chunk_end = i + WINDOW_SIZE
        current_chunk = audio_float[chunk_start:chunk_end]

        if current_chunk.size == 0:
            break

        harmonic_chunk = extractor.process_chunk(current_chunk)
        encoded = packer_compact.pack_chunk(harmonic_chunk)
        maxvalue = max(maxvalue, len(encoded) * 8)

        extracted_parameters.append(encoded)

        # report progress every 100 chunks
        if len(extracted_parameters) % 100 == 0:
            print(f"Chunk {len(extracted_parameters)}: Detected {len(harmonic_chunk)} primary frequency source(s).")

    print(f"Extraction complete. Total Chunks (Container size): {len(extracted_parameters)}")
    print(f"Max kbps is {(maxvalue*(sample_rate / WINDOW_SIZE) / 1000)} Kbps")

    #print("sample of extracted parameters from first chunk:")
    #if extracted_parameters:
    #    import pprint
    #    pprint.pprint(extracted_parameters[0])
    #else:
    #    print("No parameters extracted.")

    # --- Synthesis Loop (Generation & Overlap-Add) ---
    print("Generating and synthesizing audio...")
    for j, harmonic_chunk in enumerate(extracted_parameters):
        decoded_objects = packer_compact.unpack_chunk(harmonic_chunk)

        synthesized_chunk = generator.process_chunk(decoded_objects)
        synthesized_chunk = synthesized_chunk * generator.window

        # Overlap-Add
        start_index = j * HOP_SIZE
        end_index = start_index + WINDOW_SIZE

        add_length = min(synthesized_chunk.size, output_signal_length - start_index)

        output_signal[start_index:end_index] += synthesized_chunk[:add_length]

        if j % 100 == 0:
            print(f"Synthesized Chunk {j + 1}/{len(extracted_parameters)}")

    # Normalize and save
    max_abs_val = np.max(np.abs(output_signal))
    if max_abs_val > 1.0e-6:
        output_signal = output_signal / max_abs_val

    output_int16 = (output_signal * 32767).astype(np.int16)
    write(output_audio_path, sample_rate, output_int16)

    print(f"Synthesis complete. Output saved to {output_audio_path}")

if __name__ == '__main__':
    # --- DEMO CONFIGURATION ---

    # 1. Audio Parameters
    INPUT_FILE = r"C:\Users\sansw\Desktop\sample.wav"
    OUTPUT_FILE = r"C:\Users\sansw\Desktop\output.wav"

    # 2. Extractor/Generator Parameters
    ANALYSIS_PARAMS = {
        # Increased window size to improve frequency resolution (44100/4096 ~ 10.76 Hz per bin)
        'window_size': 4096,
        'hop_size': 512,

        # Extractor Specific Parameters
        # Setting search range to test high frequency detection
        'min_f0_freq': 20.0,
        'max_f0_freq': 6000.0,

        # Minimum magnitude threshold (0.0 to 1.0) for a peak to be considered a P0
        'peak_threshold': 0.0,

        # Max number of harmonics (P0, 2*P0, ...) to track for each P0
        'max_harmonics_per_f0': 20,

        # Upper frequency ceiling (Hz) for tracked harmonics
        'max_harmonic_freq_output': 6000.0,

        # Max number of primary frequency objects (P0s) to track per chunk
        'max_harmonic_freq_object': 5,
    }

    # --- Step 1: Create a simple test file ---
    # Create a 2-second, 440 Hz tone (A4)
    #create_dummy_wav(
    #    filepath=INPUT_FILE,
    #    duration_seconds=2.0,
    #    frequency=440,
    #    amplitude=0.5,
    #    sr=SAMPLE_RATE
    #)

    # --- Step 2: Run the Harmonic Extractor and Generator ---
    process_chunk_by_chunk(INPUT_FILE, OUTPUT_FILE, ANALYSIS_PARAMS)

    print("done!")