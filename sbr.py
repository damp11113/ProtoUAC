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
            energy_threshold_db: float = -60.0,
            flat_band_threshold: float = 0.75
    ):
        self.sample_rate = sample_rate
        self.floor_db = floor_db
        self.pre_echo_threshold = pre_echo_threshold
        self.energy_threshold_db = energy_threshold_db
        self.flat_band_threshold = flat_band_threshold

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

    def analyze(self, mono: np.ndarray, infloat=False, adaptive=False) -> Tuple[List[float], List[bool], bool]:
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
        n_fft = len(audio_float)
        mono_fft = np.fft.rfft(audio_float)

        # Get frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)

        # Calculate magnitude spectrum
        mono_mag = np.abs(mono_fft)

        # Extract energy for each band
        band_energies = []
        shouldUseNoises = []
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

                if adaptive:
                    band_power = np.abs(band_mag) ** 2
                    # Avoid log(0)
                    band_power = np.maximum(band_power, 1e-12)

                    # Compute spectral flatness
                    geo_mean = np.exp(np.mean(np.log(band_power)))
                    arith_mean = np.mean(band_power)
                    sfm = geo_mean / arith_mean

                    # Compute total power in that band (for energy check)
                    band_energy_db = 10 * np.log10(np.mean(band_power))

                    # Decide if noise should be used based on spectral flatness and energy
                    if sfm > self.flat_band_threshold and band_energy_db > (self.energy_threshold_db // 3):
                        #print(sfm, band_energy_db)
                        isNoise = True
                    else:
                        isNoise = False
                else:
                    isNoise = False
            else:
                energy_db = self.floor_db
                isNoise = False

            band_energies.append(energy_db)

            shouldUseNoises.append(isNoise)

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
        return band_energies, shouldUseNoises, is_transient

class SBRDecoder:
    """SBR Decoder with hybrid noise/transposition synthesis per band."""

    def __init__(
            self,
            sample_rate: int,
            min_freq: float,
            max_freq: float,
            freq_points: int,
            chunk_size: int = 256,
            overlap: float = 0.5,
            noise_gain: float = 0.1
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.hop_size = int(chunk_size * (1 - overlap))
        self.noise_gain = noise_gain

        self.set_freq(min_freq, max_freq, freq_points)

        self.prev_hf_signal = None
        self.window = self._create_window(chunk_size)

    def _create_window(self, size: int) -> np.ndarray:
        """Create Hann window for smooth overlap-add."""
        return np.hanning(size).astype(np.float32)

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        """Dynamically set frequency band parameters in realtime."""
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        log_min = np.log2(min_freq)
        log_max = np.log2(max_freq)
        self.target_freqs = 2 ** np.linspace(log_min, log_max, freq_points)

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

    def _generate_hybrid_chunk(
            self,
            baseband_chunk: np.ndarray,
            band_energies: List[float],
            should_use_noises: List[bool],
            energy_modulation: float = 1.0
    ) -> np.ndarray:
        """Generate HF content using hybrid noise/transposition per band."""
        chunk_length = len(baseband_chunk)

        # Prepare baseband transposition
        baseband_fft = np.fft.rfft(baseband_chunk)
        freqs = np.fft.rfftfreq(chunk_length, 1 / self.sample_rate)

        transposed_fft = np.zeros_like(baseband_fft, dtype=np.complex128)

        hf_bandwidth = self.max_freq - self.min_freq
        transposition_order = int(np.ceil(self.min_freq / hf_bandwidth))

        # Harmonic transposition for non-noise bands
        for order in range(transposition_order, transposition_order + 3):
            shift_factor = order

            for i, freq in enumerate(freqs):
                if freq > 0 and freq < hf_bandwidth:
                    shifted_freq = freq * shift_factor

                    if self.min_freq <= shifted_freq <= self.max_freq:
                        target_idx = np.argmin(np.abs(freqs - shifted_freq))

                        if target_idx < len(transposed_fft):
                            phase_rand = np.exp(1j * np.random.uniform(-0.5, 0.5))
                            transposed_fft[target_idx] += baseband_fft[i] * phase_rand * (1.0 / shift_factor)

        # Prepare noise source
        noise = np.random.randn(chunk_length).astype(np.float32)
        noise_fft = np.fft.rfft(noise)

        # Hybrid spectrum: combine transposition and noise per band
        shaped_fft = np.zeros_like(baseband_fft, dtype=np.complex128)

        for target_freq, bandwidth, energy_db, use_noise in zip(
                self.target_freqs, self.bandwidths, band_energies, should_use_noises
        ):
            # Apply temporal modulation to energy
            modulated_energy = energy_db + 20 * np.log10(energy_modulation) if energy_modulation > 0 else -80.0

            if modulated_energy <= -70.0:
                continue

            freq_low = target_freq - bandwidth / 2
            freq_high = target_freq + bandwidth / 2
            mask = (freqs >= freq_low) & (freqs <= freq_high)

            if not np.any(mask):
                continue

            sigma = bandwidth / 4
            band_freqs = freqs[mask]
            gaussian = np.exp(-0.5 * ((band_freqs - target_freq) / sigma) ** 2)

            if use_noise:
                # Noise-based synthesis
                energy_linear = 10 ** (modulated_energy / 20.0)
                shaped_fft[mask] += noise_fft[mask] * energy_linear * gaussian * self.noise_gain
            else:
                # Transposition-based synthesis with energy shaping
                current_band = transposed_fft[mask]
                current_rms = np.sqrt(np.mean(np.abs(current_band) ** 2))

                if current_rms > 1e-10:
                    target_linear = 10 ** (modulated_energy / 20.0)
                    gain = target_linear / current_rms
                    shaped_fft[mask] += transposed_fft[mask] * (1.0 + (gain - 1.0) * gaussian * 0.7)
                else:
                    # Fallback to noise if transposition is silent
                    energy_linear = 10 ** (modulated_energy / 20.0)
                    shaped_fft[mask] += noise_fft[mask] * energy_linear * gaussian * self.noise_gain

        hf_chunk = np.fft.irfft(shaped_fft, n=chunk_length)
        return hf_chunk

    def generate(
            self,
            baseband_mono: np.ndarray,
            band_energies: List[float],
            is_transient: bool,
            should_use_noises: List[bool],
            infloat: bool = False,
            use_chunks: bool = True
    ) -> np.ndarray:
        """
        Generate high-frequency content with hybrid synthesis.

        Args:
            baseband_mono: Lossy baseband signal (mono)
            band_energies: Target energy envelope per band
            is_transient: Whether current frame is transient
            should_use_noises: Per-band flags for noise (True) vs transposition (False)
            infloat: If True, input is already float32
            use_chunks: If True, use sub-chunk processing

        Returns:
            High-frequency signal as float32
        """
        if infloat:
            audio_float = baseband_mono
        else:
            audio_float = baseband_mono.astype(np.float32) / 32768.0

        frame_length = len(audio_float)

        if not use_chunks or is_transient:
            return self._generate_single_frame(audio_float, band_energies, is_transient, should_use_noises)

        # Sub-chunk processing with overlap-add
        output = np.zeros(frame_length, dtype=np.float32)
        window_sum = np.zeros(frame_length, dtype=np.float32)

        num_chunks = (frame_length - self.chunk_size) // self.hop_size + 1

        for i in range(num_chunks):
            start = i * self.hop_size
            end = start + self.chunk_size

            if end > frame_length:
                break

            # Extract baseband chunk
            baseband_chunk = audio_float[start:end]

            # Calculate local energy modulation from baseband dynamics
            local_rms = np.sqrt(np.mean(baseband_chunk ** 2))
            global_rms = np.sqrt(np.mean(audio_float ** 2))

            if global_rms > 1e-8:
                energy_modulation = np.clip(local_rms / global_rms, 0.5, 2.0)
            else:
                energy_modulation = 1.0

            # Generate hybrid HF chunk
            hf_chunk = self._generate_hybrid_chunk(
                baseband_chunk,
                band_energies,
                should_use_noises,
                energy_modulation
            )

            # Apply window
            windowed_chunk = hf_chunk * self.window

            # Overlap-add
            output[start:end] += windowed_chunk
            window_sum[start:end] += self.window

        # Normalize
        mask = window_sum > 1e-8
        output[mask] /= window_sum[mask]

        # Normalize to prevent overload
        rms = np.sqrt(np.mean(output ** 2))
        if rms > 0.15:
            output = output * (0.15 / rms)

        # Crossfade with previous
        if self.prev_hf_signal is not None:
            crossfade_len = min(128, frame_length // 8)
            if len(self.prev_hf_signal) >= crossfade_len:
                fade = np.linspace(0, 1, crossfade_len)
                output[:crossfade_len] = (
                        output[:crossfade_len] * fade +
                        self.prev_hf_signal[-crossfade_len:] * (1 - fade)
                )

        self.prev_hf_signal = output.copy()

        return output.astype(np.float32)

    def _generate_single_frame(
            self,
            audio_float: np.ndarray,
            band_energies: List[float],
            is_transient: bool,
            should_use_noises: List[bool]
    ) -> np.ndarray:
        """Single-frame generation with hybrid synthesis."""
        frame_length = len(audio_float)

        # Prepare baseband transposition
        baseband_fft = np.fft.rfft(audio_float)
        freqs = np.fft.rfftfreq(frame_length, 1 / self.sample_rate)

        transposed_fft = np.zeros_like(baseband_fft, dtype=np.complex128)

        hf_bandwidth = self.max_freq - self.min_freq
        transposition_order = int(np.ceil(self.min_freq / hf_bandwidth))

        for order in range(transposition_order, transposition_order + 3):
            shift_factor = order

            for i, freq in enumerate(freqs):
                if freq > 0 and freq < hf_bandwidth:
                    shifted_freq = freq * shift_factor

                    if self.min_freq <= shifted_freq <= self.max_freq:
                        target_idx = np.argmin(np.abs(freqs - shifted_freq))

                        if target_idx < len(transposed_fft):
                            phase_rand = np.exp(1j * np.random.uniform(-0.5, 0.5))
                            transposed_fft[target_idx] += baseband_fft[i] * phase_rand * (1.0 / shift_factor)

        # Prepare noise
        noise = np.random.randn(frame_length).astype(np.float32)
        noise_fft = np.fft.rfft(noise)

        # Hybrid shaping
        shaped_fft = np.zeros_like(baseband_fft, dtype=np.complex128)

        for target_freq, bandwidth, energy_db, use_noise in zip(
                self.target_freqs, self.bandwidths, band_energies, should_use_noises
        ):
            if energy_db <= -70.0:
                continue

            freq_low = target_freq - bandwidth / 2
            freq_high = target_freq + bandwidth / 2
            mask = (freqs >= freq_low) & (freqs <= freq_high)

            if not np.any(mask):
                continue

            sigma = bandwidth / 4
            band_freqs = freqs[mask]
            gaussian = np.exp(-0.5 * ((band_freqs - target_freq) / sigma) ** 2)

            if use_noise:
                # Noise-based synthesis
                energy_linear = 10 ** (energy_db / 20.0)
                shaped_fft[mask] += noise_fft[mask] * energy_linear * gaussian * self.noise_gain
            else:
                # Transposition-based synthesis
                current_band = transposed_fft[mask]
                current_rms = np.sqrt(np.mean(np.abs(current_band) ** 2))

                if current_rms > 1e-10:
                    target_linear = 10 ** (energy_db / 20.0)
                    gain = target_linear / current_rms
                    shaped_fft[mask] += transposed_fft[mask] * (1.0 + (gain - 1.0) * gaussian * 0.7)
                else:
                    # Fallback to noise if transposition is silent
                    energy_linear = 10 ** (energy_db / 20.0)
                    shaped_fft[mask] += noise_fft[mask] * energy_linear * gaussian * self.noise_gain

        hf_signal = np.fft.irfft(shaped_fft, n=frame_length)

        rms = np.sqrt(np.mean(hf_signal ** 2))
        if rms > 0.15:
            hf_signal = hf_signal * (0.15 / rms)

        if self.prev_hf_signal is not None and not is_transient:
            crossfade_len = min(128, frame_length // 8)
            if len(self.prev_hf_signal) >= crossfade_len:
                fade = np.linspace(0, 1, crossfade_len)
                hf_signal[:crossfade_len] = (
                        hf_signal[:crossfade_len] * fade +
                        self.prev_hf_signal[-crossfade_len:] * (1 - fade)
                )

        if is_transient:
            attack_len = min(32, frame_length // 16)
            attack_curve = np.linspace(0, 1, attack_len) ** 2
            hf_signal[:attack_len] *= attack_curve

        self.prev_hf_signal = hf_signal.copy()

        return hf_signal.astype(np.float32)