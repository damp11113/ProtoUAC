from scipy import signal
import numpy as np
from typing import List, Tuple, Optional


class SBREncoder:
    """SBR Encoder with pre-echo detection and control."""

    def __init__(
            self,
            sample_rate: int,
            min_freq: float,
            max_freq: float,
            freq_points: int,
            floor_db: float = -75.0,
            pre_echo_threshold: float = 10.0,
            energy_threshold_db: float = -60.0
    ):
        self.sample_rate = sample_rate
        self.floor_db = floor_db
        self.pre_echo_threshold = pre_echo_threshold
        self.energy_threshold_db = energy_threshold_db

        # Initialize frequency bands
        self.set_freq(min_freq, max_freq, freq_points)

        self.prev_energy = None
        self.prev_band_energies = None

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        """
        Dynamically set frequency band parameters in realtime.

        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
            freq_points: Number of frequency bands
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        # Generate logarithmic frequency bands
        log_min = np.log2(min_freq)
        log_max = np.log2(max_freq)
        self.target_freqs = 2 ** np.linspace(log_min, log_max, freq_points)

        # Calculate bandwidths
        self.bandwidths = []
        for i in range(len(self.target_freqs)):
            if i == 0:
                if freq_points > 1:
                    bw = (self.target_freqs[i + 1] - self.target_freqs[i])
                else:
                    bw = max_freq - min_freq
            elif i == len(self.target_freqs) - 1:
                bw = self.target_freqs[i] - self.target_freqs[i - 1]
            else:
                bw = (self.target_freqs[i + 1] - self.target_freqs[i - 1]) / 2
            self.bandwidths.append(bw)

        # Reset previous band energies when frequency configuration changes
        self.prev_band_energies = None

    def detect_transient(self, current_energy: float, prev_energy: Optional[float]) -> bool:
        """Detect transient/attack by comparing energy with previous frame."""
        if prev_energy is None:
            return False
        energy_increase = current_energy - prev_energy
        return energy_increase > self.pre_echo_threshold

    def analyze(self, mono: np.ndarray, infloat=False) -> Tuple[List[float], bool]:
        """
        Analyze energy in frequency bands.
        Returns: (band_energies, is_transient)
        """
        # Normalize int16 to float
        if infloat:
            audio_float = mono
        else:
            audio_float = mono.astype(np.float32) / 32768.0

        # Calculate overall energy for transient detection
        current_energy = 20 * np.log10(np.sqrt(np.mean(audio_float ** 2)) + 1e-10)
        is_transient = self.detect_transient(current_energy, self.prev_energy)
        self.prev_energy = current_energy

        # Apply window to reduce spectral leakage
        window = np.hanning(len(audio_float))
        audio_windowed = audio_float * window

        # Calculate FFT
        n_fft = len(audio_windowed)
        mono_fft = np.fft.rfft(audio_windowed)

        # Get frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)

        # Calculate magnitude spectrum
        mono_mag = np.abs(mono_fft)

        # Extract energy for each band
        band_energies = []
        for target_freq, bandwidth in zip(self.target_freqs, self.bandwidths):
            freq_low = target_freq - bandwidth / 2
            freq_high = target_freq + bandwidth / 2
            mask = (freqs >= freq_low) & (freqs <= freq_high)

            if np.any(mask):
                # Get mean energy in this band
                band_mag = mono_mag[mask]
                mean_energy = np.mean(band_mag)

                # Convert to dB
                energy_db = 20 * np.log10(mean_energy + 1e-10)

                # Apply energy threshold
                if energy_db < self.energy_threshold_db:
                    energy_db = self.floor_db
            else:
                energy_db = self.floor_db

            band_energies.append(energy_db)

        # Reduced temporal smoothing - only for non-transients
        if self.prev_band_energies is not None and not is_transient:
            # Only smooth if band count matches
            if len(self.prev_band_energies) == len(band_energies):
                smoothing_factor = 0.3  # Reduced from 0.5
                band_energies = [
                    prev * smoothing_factor + curr * (1 - smoothing_factor)
                    for prev, curr in zip(self.prev_band_energies, band_energies)
                ]

        self.prev_band_energies = band_energies.copy()
        return band_energies, is_transient

class SBRDecoder:
    """SBR Decoder with pre-echo aware synthesis."""

    def __init__(
            self,
            sample_rate: int,
            min_freq: float,
            max_freq: float,
            freq_points: int
    ):
        self.sample_rate = sample_rate

        # Initialize frequency bands
        self.set_freq(min_freq, max_freq, freq_points)

        self.prev_hf_signal = None

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        """
        Dynamically set frequency band parameters in realtime.

        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
            freq_points: Number of frequency bands
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        # Generate logarithmic frequency bands
        log_min = np.log2(min_freq)
        log_max = np.log2(max_freq)
        self.target_freqs = 2 ** np.linspace(log_min, log_max, freq_points)

        # Calculate bandwidths
        self.bandwidths = []
        for i in range(len(self.target_freqs)):
            if i == 0:
                if freq_points > 1:
                    bw = self.target_freqs[i + 1] - self.target_freqs[i]
                else:
                    bw = max_freq - min_freq
            elif i == len(self.target_freqs) - 1:
                bw = self.target_freqs[i] - self.target_freqs[i - 1]
            else:
                bw = (self.target_freqs[i + 1] - self.target_freqs[i - 1]) / 2
            self.bandwidths.append(bw)

        # Create smooth filters for each band to reduce ringing
        self.band_filters = self._create_band_filters()

    def _create_band_filters(self):
        """Create smooth gaussian-like filters for each band."""
        filters = []
        for target_freq, bandwidth in zip(self.target_freqs, self.bandwidths):
            filters.append((target_freq, bandwidth))
        return filters

    def generate(
            self,
            frame_length: int,
            band_energies: List[float],
            is_transient: bool
    ) -> np.ndarray:
        """
        Generate high-frequency content from energy envelope.
        """
        # Generate white noise
        noise = np.random.randn(frame_length).astype(np.float32)

        # FFT of noise
        noise_fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(frame_length, 1 / self.sample_rate)

        # Create shaped spectrum with smooth filters
        shaped_fft = np.zeros_like(noise_fft, dtype=np.complex128)

        for target_freq, bandwidth, energy_db in zip(self.target_freqs, self.bandwidths, band_energies):
            # Skip silent bands
            if energy_db <= -70.0:
                continue

            # Convert energy from dB to linear
            energy_linear = 10 ** (energy_db / 20.0)

            # Create smooth gaussian envelope for this band
            freq_low = target_freq - bandwidth / 2
            freq_high = target_freq + bandwidth / 2

            # Use gaussian weighting instead of hard mask
            sigma = bandwidth / 4  # Controls smoothness
            gaussian = np.exp(-0.5 * ((freqs - target_freq) / sigma) ** 2)

            # Only apply within reasonable range
            mask = (freqs >= freq_low) & (freqs <= freq_high)

            if np.any(mask):
                shaped_fft[mask] += noise_fft[mask] * energy_linear * gaussian[mask]

        # Inverse FFT to get time-domain signal
        hf_signal = np.fft.irfft(shaped_fft, n=frame_length)

        # Simple crossfade for non-transients
        if self.prev_hf_signal is not None and not is_transient:
            crossfade_len = min(128, frame_length // 8)  # Shorter crossfade
            if len(self.prev_hf_signal) >= crossfade_len:
                fade = np.linspace(0, 1, crossfade_len)
                hf_signal[:crossfade_len] = (
                        hf_signal[:crossfade_len] * fade +
                        self.prev_hf_signal[-crossfade_len:] * (1 - fade)
                )

        # For transients, apply quick fade-in
        if is_transient:
            attack_len = min(32, frame_length // 16)
            attack_curve = np.linspace(0, 1, attack_len) ** 2
            hf_signal[:attack_len] *= attack_curve

        # Store for next frame
        self.prev_hf_signal = hf_signal.copy()

        # NO per-frame normalization - maintain consistent level
        # Just apply a fixed scale factor
        hf_signal = hf_signal * 0.3

        return hf_signal.astype(np.float32)


class SBRDecoderHR:
    """SBR Decoder with transposition-based synthesis."""

    def __init__(
            self,
            sample_rate: int,
            min_freq: float,
            max_freq: float,
            freq_points: int
    ):
        self.sample_rate = sample_rate

        # Initialize frequency bands
        self.set_freq(min_freq, max_freq, freq_points)

        self.prev_hf_signal = None

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        """
        Dynamically set frequency band parameters in realtime.

        Args:
            min_freq: Minimum frequency in Hz (start of HF region)
            max_freq: Maximum frequency in Hz
            freq_points: Number of frequency bands
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        # Generate logarithmic frequency bands
        log_min = np.log2(min_freq)
        log_max = np.log2(max_freq)
        self.target_freqs = 2 ** np.linspace(log_min, log_max, freq_points)

        # Calculate bandwidths
        self.bandwidths = []
        for i in range(len(self.target_freqs)):
            if i == 0:
                if freq_points > 1:
                    bw = self.target_freqs[i + 1] - self.target_freqs[i]
                else:
                    bw = max_freq - min_freq
            elif i == len(self.target_freqs) - 1:
                bw = self.target_freqs[i] - self.target_freqs[i - 1]
            else:
                bw = (self.target_freqs[i + 1] - self.target_freqs[i - 1]) / 2
            self.bandwidths.append(bw)

    def generate(
            self,
            baseband_mono: np.ndarray,
            band_energies: List[float],
            is_transient: bool,
            infloat: bool = False
    ) -> np.ndarray:
        """
        Generate high-frequency content from baseband signal via transposition.

        Args:
            baseband_mono: Lossy baseband signal (mono)
            band_energies: Target energy envelope from encoder
            is_transient: Whether current frame is transient
            infloat: If True, input is already float32, else int16

        Returns:
            High-frequency signal as float32
        """
        # Normalize int16 to float if needed
        if infloat:
            audio_float = baseband_mono
        else:
            audio_float = baseband_mono.astype(np.float32) / 32768.0

        frame_length = len(audio_float)

        # Step 1: FFT of baseband signal (no window)
        baseband_fft = np.fft.rfft(audio_float)
        freqs = np.fft.rfftfreq(frame_length, 1 / self.sample_rate)

        # Step 2: Transpose baseband to HF region
        # The baseband contains frequencies from 0 to min_freq
        # We need to transpose it to fill min_freq to max_freq
        transposed_fft = np.zeros_like(baseband_fft, dtype=np.complex128)

        # Calculate transposition factor (typically 2x, 3x, or 4x)
        # For 8kHz baseband to 8-14kHz HF: factor ~1.0-1.75x
        baseband_bandwidth = self.min_freq
        hf_bandwidth = self.max_freq - self.min_freq

        # Use harmonic transposition - shift baseband by integer multiples
        transposition_order = int(np.ceil(self.min_freq / hf_bandwidth))

        # Apply vectorized transposition
        for order in range(transposition_order, transposition_order + 3):
            shift_factor = order

            for i, freq in enumerate(freqs):
                if freq > 0 and freq < hf_bandwidth:
                    shifted_freq = freq * shift_factor

                    if self.min_freq <= shifted_freq <= self.max_freq:
                        # Find target bin
                        target_idx = np.argmin(np.abs(freqs - shifted_freq))

                        if target_idx < len(transposed_fft):
                            # Add with slight phase randomization (faster)
                            phase_rand = np.exp(1j * np.random.uniform(-0.5, 0.5))
                            transposed_fft[target_idx] += baseband_fft[i] * phase_rand * (1.0 / shift_factor)

        # Step 3: Apply energy adjustment from encoder
        shaped_fft = transposed_fft.copy()

        for target_freq, bandwidth, energy_db in zip(self.target_freqs, self.bandwidths, band_energies):
            # Skip silent bands
            if energy_db <= -70.0:
                freq_low = target_freq - bandwidth / 2
                freq_high = target_freq + bandwidth / 2
                mask = (freqs >= freq_low) & (freqs <= freq_high)
                shaped_fft[mask] = 0
                continue

            # Define band range
            freq_low = target_freq - bandwidth / 2
            freq_high = target_freq + bandwidth / 2
            mask = (freqs >= freq_low) & (freqs <= freq_high)

            if np.any(mask):
                # Calculate current RMS energy in band
                current_band = shaped_fft[mask]
                current_rms = np.sqrt(np.mean(np.abs(current_band) ** 2))

                if current_rms > 1e-10:
                    # Convert target energy from dB to linear
                    target_linear = 10 ** (energy_db / 20.0)

                    # Calculate gain adjustment
                    gain = target_linear / current_rms

                    # Apply gain with smooth gaussian weighting
                    sigma = bandwidth / 4
                    band_freqs = freqs[mask]
                    gaussian = np.exp(-0.5 * ((band_freqs - target_freq) / sigma) ** 2)

                    # Smooth adjustment
                    shaped_fft[mask] *= (1.0 + (gain - 1.0) * gaussian * 0.7)

        # Step 4: Inverse FFT
        hf_signal = np.fft.irfft(shaped_fft, n=frame_length)

        # Step 5: Normalize to prevent overload
        # Calculate RMS and apply gentle limiting
        rms = np.sqrt(np.mean(hf_signal ** 2))
        if rms > 0.15:  # Prevent overload
            hf_signal = hf_signal * (0.15 / rms)

        # Step 6: Temporal smoothing for non-transients
        if self.prev_hf_signal is not None and not is_transient:
            crossfade_len = min(128, frame_length // 8)
            if len(self.prev_hf_signal) >= crossfade_len:
                fade = np.linspace(0, 1, crossfade_len)
                hf_signal[:crossfade_len] = (
                        hf_signal[:crossfade_len] * fade +
                        self.prev_hf_signal[-crossfade_len:] * (1 - fade)
                )

        # For transients, apply quick fade-in
        if is_transient:
            attack_len = min(32, frame_length // 16)
            attack_curve = np.linspace(0, 1, attack_len) ** 2
            hf_signal[:attack_len] *= attack_curve

        # Store for next frame
        self.prev_hf_signal = hf_signal.copy()

        # Final scaling (reduced from 0.3 to prevent overload)
        hf_signal = hf_signal

        return hf_signal.astype(np.float32)