import numpy as np
from typing import List, Tuple

class PSEncoder:
    def __init__(
            self,
            sample_rate: int,
            min_freq: float = 20.0,
            max_freq: float = 20000.0,
            freq_points: int = 32,
            floor_db: float = -75.0,
            use_grouping: bool = False,
            log_scale: bool = True
    ):
        """Initialize the stereo audio analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            min_freq: Minimum frequency to analyze in Hz
            max_freq: Maximum frequency to analyze in Hz
            freq_points: Number of frequency points to sample
            floor_db: Noise floor in dB (signals below this are ignored)
            use_grouping: If True, average over frequency bands
            log_scale: If True, use logarithmic frequency spacing (default)
        """
        self.sample_rate = sample_rate
        self.floor_db = floor_db
        self.use_grouping = use_grouping
        self.log_scale = log_scale

        # Pre-calculate target frequencies
        self.set_freq(min_freq, max_freq, freq_points)

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        if self.log_scale:
            self.target_freqs = np.logspace(
                np.log10(min_freq),
                np.log10(max_freq),
                freq_points
            )
        else:
            self.target_freqs = np.linspace(min_freq, max_freq, freq_points)

    def analyze(
            self,
            audio_data: np.ndarray,
            infloat=False
    ) -> List[Tuple[float, float, float, float]]:
        """Analyze stereo audio and extract IID (pan), IPD, and IC per frequency band.

        Args:
            audio_data: Stereo audio data with shape [samples, 2]

        Returns:
            List of tuples: [(freq, pan, ipd, ic), ...]
            - freq: frequency in Hz
            - pan: Inter-aural Intensity Difference (IID) as pan value [-1, 1]
            - ipd: Inter-aural Phase Difference in radians [-π, π]
            - ic: Inter-channel Coherence [0, 1]
        """
        if audio_data.ndim != 2 or audio_data.shape[1] != 2:
            raise ValueError("Audio must be stereo (shape: [samples, 2])")

        # Normalize int16 to float
        if infloat:
            audio_float = audio_data
        else:
            audio_float = audio_data.astype(np.float32) / 32768.0

        # Separate channels
        left = audio_float[:, 0]
        right = audio_float[:, 1]

        # Calculate FFT for both channels
        n_fft = len(left)
        left_fft = np.fft.rfft(left)
        right_fft = np.fft.rfft(right)

        # Get frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)

        # Calculate magnitude for each channel
        left_mag = np.abs(left_fft)
        right_mag = np.abs(right_fft)

        # Calculate combined magnitude for floor detection
        combined_mag = (left_mag + right_mag) / 2

        # Convert floor from dB to linear scale
        floor_linear = 10 ** (self.floor_db / 20.0)

        # Calculate bandwidth for grouping
        if self.use_grouping and self.freq_points > 1:
            if self.log_scale:
                # For log scale, bandwidth is proportional to frequency
                bandwidths = self._calculate_log_bandwidths()
            else:
                bandwidth = (self.max_freq - self.min_freq) / self.freq_points
                bandwidths = [bandwidth] * self.freq_points
        else:
            bandwidths = [0] * self.freq_points

        # Analyze each frequency band
        results = []
        for target_freq, bandwidth in zip(self.target_freqs, bandwidths):
            if self.use_grouping and bandwidth > 0:
                freq_low = target_freq - bandwidth / 2
                freq_high = target_freq + bandwidth / 2
                mask = (freqs >= freq_low) & (freqs <= freq_high)

                if np.any(mask):
                    pan, ipd, ic = self._analyze_band(
                        left_fft[mask], right_fft[mask],
                        left_mag[mask], right_mag[mask],
                        combined_mag[mask], floor_linear
                    )
                else:
                    pan, ipd, ic = 0.0, 0.0, 0.0
            else:
                idx = np.argmin(np.abs(freqs - target_freq))
                pan, ipd, ic = self._analyze_bin(
                    left_fft[idx], right_fft[idx],
                    left_mag[idx], right_mag[idx],
                    combined_mag[idx], floor_linear
                )

            results.append((float(target_freq), float(pan), float(ipd), float(ic)))

        return results

    def _calculate_log_bandwidths(self) -> List[float]:
        """Calculate bandwidths for logarithmic frequency spacing."""
        bandwidths = []
        log_freqs = np.log10(self.target_freqs)

        for i in range(len(log_freqs)):
            if i == 0:
                # First band: extend to halfway to next band
                log_width = (log_freqs[i + 1] - log_freqs[i])
            elif i == len(log_freqs) - 1:
                # Last band: extend to halfway from previous band
                log_width = (log_freqs[i] - log_freqs[i - 1])
            else:
                # Middle bands: halfway to neighbors on each side
                log_width = (log_freqs[i + 1] - log_freqs[i - 1]) / 2

            # Convert log width to linear bandwidth
            freq_low = 10 ** (log_freqs[i] - log_width / 2)
            freq_high = 10 ** (log_freqs[i] + log_width / 2)
            bandwidths.append(freq_high - freq_low)

        return bandwidths

    def _analyze_band(
            self,
            left_fft: np.ndarray,
            right_fft: np.ndarray,
            left_mag: np.ndarray,
            right_mag: np.ndarray,
            combined_mag: np.ndarray,
            floor_linear: float
    ) -> Tuple[float, float, float]:
        """Analyze a frequency band and return pan, ipd, ic."""
        band_left_mag = np.mean(left_mag)
        band_right_mag = np.mean(right_mag)
        band_combined_mag = np.mean(combined_mag)

        if band_combined_mag < floor_linear:
            return 0.0, 0.0, 0.0

        # Calculate IID (pan)
        total = band_left_mag + band_right_mag
        if total > 1e-10:
            pan_value = (band_left_mag - band_right_mag) / total
        else:
            pan_value = 0.0
        pan_value = np.clip(pan_value, -1.0, 1.0)

        # IPD - Phase difference (averaged over band)
        left_phase = np.angle(left_fft)
        right_phase = np.angle(right_fft)
        phase_diff = left_phase - right_phase
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Weighted average by magnitude
        weights = np.abs(left_fft) + np.abs(right_fft)
        if np.sum(weights) > 1e-10:
            ipd_value = np.average(phase_diff, weights=weights)
        else:
            ipd_value = 0.0

        # IC - Inter-channel Coherence
        cross_power = np.mean(left_fft * np.conj(right_fft))
        left_power = np.mean(np.abs(left_fft) ** 2)
        right_power = np.mean(np.abs(right_fft) ** 2)

        if left_power > 1e-10 and right_power > 1e-10:
            ic_value = np.abs(cross_power) / np.sqrt(left_power * right_power)
            ic_value = np.clip(ic_value, 0.0, 1.0)
        else:
            ic_value = 0.0

        return pan_value, ipd_value, ic_value

    def _analyze_bin(
            self,
            left_fft: complex,
            right_fft: complex,
            left_mag: float,
            right_mag: float,
            combined_mag: float,
            floor_linear: float
    ) -> Tuple[float, float, float]:
        """Analyze a single frequency bin and return pan, ipd, ic."""
        if combined_mag < floor_linear:
            return 0.0, 0.0, 0.0

        # Calculate IID (pan)
        total = left_mag + right_mag
        if total > 1e-10:
            pan_value = (left_mag - right_mag) / total
        else:
            pan_value = 0.0
        pan_value = np.clip(pan_value, -1.0, 1.0)

        # IPD - Phase difference
        left_phase = np.angle(left_fft)
        right_phase = np.angle(right_fft)
        phase_diff = left_phase - right_phase
        ipd_value = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # IC - Inter-channel Coherence
        cross_power = left_fft * np.conj(right_fft)
        left_power = np.abs(left_fft) ** 2
        right_power = np.abs(right_fft) ** 2

        if left_power > 1e-10 and right_power > 1e-10:
            ic_value = np.abs(cross_power) / np.sqrt(left_power * right_power)
            ic_value = np.clip(ic_value, 0.0, 1.0)
        else:
            ic_value = 0.0

        return pan_value, ipd_value, ic_value

class PSDecoder:
    def __init__(
            self,
            sample_rate: int,
            min_freq: float = 20.0,
            max_freq: float = 20000.0,
            freq_points: int = 32,
            use_grouping: bool = False,
            log_scale: bool = True,
            stereo_width: float = 1.0
    ):
        """Initialize the stereo audio analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            min_freq: Minimum frequency to analyze in Hz
            max_freq: Maximum frequency to analyze in Hz
            freq_points: Number of frequency points to sample
            floor_db: Noise floor in dB (signals below this are ignored)
            use_grouping: If True, average over frequency bands
            log_scale: If True, use logarithmic frequency spacing (default)
            stereo_width: Stereo width scaling factor
        """
        self.sample_rate = sample_rate
        self.use_grouping = use_grouping
        self.log_scale = log_scale
        self.stereo_width = stereo_width

        # Pre-calculate target frequencies
        self.set_freq(min_freq, max_freq, freq_points)

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        if self.log_scale:
            self.target_freqs = np.logspace(
                np.log10(min_freq),
                np.log10(max_freq),
                freq_points
            )
        else:
            self.target_freqs = np.linspace(min_freq, max_freq, freq_points)

    def _apply_stereo_width(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stereo width expansion using Mid-Side processing.
        This is the standard algorithm used in DAWs.
        """
        # Convert L/R to Mid-Side
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        
        # Apply width to side signal
        side_widened = side * self.stereo_width
        
        # Convert back to L/R
        left_out = mid + side_widened
        right_out = mid - side_widened
        return left_out, right_out
        
    def apply(
            self,
            mono_audio: np.ndarray,
            pan_values: List[float],
            ipd_values: List[float],
            ic_values: List[bool]
    ) -> np.ndarray:
        """Apply frequency-dependent IID (pan), IPD (phase), and IC (coherence) to mono audio.

        Args:
            mono_audio: Mono audio data
            pan_values: Pan values for each frequency point [-1, 1]
            ipd_values: Phase difference values for each frequency point [radians]
            ic_values: Coherence flags for each frequency point [bool]

        Returns:
            Stereo audio data with shape [samples, 2]
        """
        # Normalize if int16
        if mono_audio.dtype == np.int16:
            mono_float = mono_audio.astype(np.float32) / 32768.0
            return_int16 = True
        else:
            mono_float = mono_audio.astype(np.float32)
            return_int16 = False

        # FFT
        mono_fft = np.fft.rfft(mono_float)
        freqs = np.fft.rfftfreq(len(mono_float), 1 / self.sample_rate)

        # Calculate bandwidth for grouping
        if self.use_grouping and self.freq_points > 1:
            if self.log_scale:
                bandwidths = self._calculate_log_bandwidths()
            else:
                bandwidth = (self.max_freq - self.min_freq) / self.freq_points
                bandwidths = [bandwidth] * self.freq_points
        else:
            bandwidths = [0] * self.freq_points

        # Initialize gain and phase shift arrays
        left_gain = np.ones(len(freqs), dtype=np.float32)
        right_gain = np.ones(len(freqs), dtype=np.float32)
        right_phase_shift = np.ones(len(freqs), dtype=np.complex64)

        # Apply stereo parameters to each frequency band
        for freq, pan, ipd, ic, bandwidth in zip(
                self.target_freqs, pan_values, ipd_values, ic_values, bandwidths
        ):
            # Constant power panning
            angle = (-pan + 1.0) * np.pi / 4.0
            left_g = np.cos(angle)
            right_g = np.sin(angle)

            # If IC is False, blend toward mono
            if not ic:
                left_g = right_g = (left_g + right_g) / 2.0
                ipd = 0.0  # remove phase difference

            if self.use_grouping and bandwidth > 0:
                freq_low = freq - bandwidth / 2
                freq_high = freq + bandwidth / 2
                mask = (freqs >= freq_low) & (freqs <= freq_high)
                left_gain[mask] = left_g
                right_gain[mask] = right_g
                right_phase_shift[mask] *= np.exp(1j * -ipd)
            else:
                idx = np.argmin(np.abs(freqs - freq))
                left_gain[idx] = left_g
                right_gain[idx] = right_g
                right_phase_shift[idx] *= np.exp(1j * -ipd)

        # Apply gains and phase
        left_fft = mono_fft * left_gain
        right_fft = mono_fft * right_gain * right_phase_shift

        left = np.fft.irfft(left_fft, n=len(mono_float))
        right = np.fft.irfft(right_fft, n=len(mono_float))

        # Apply stereo width
        left, right = self._apply_stereo_width(left, right)

        stereo = np.column_stack([left, right])

        if return_int16:
            stereo = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

        return stereo

    def _calculate_log_bandwidths(self) -> List[float]:
        """Calculate bandwidths for logarithmic frequency spacing."""
        bandwidths = []
        log_freqs = np.log10(self.target_freqs)

        for i in range(len(log_freqs)):
            if i == 0:
                # First band: extend to halfway to next band
                log_width = (log_freqs[i + 1] - log_freqs[i])
            elif i == len(log_freqs) - 1:
                # Last band: extend to halfway from previous band
                log_width = (log_freqs[i] - log_freqs[i - 1])
            else:
                # Middle bands: halfway to neighbors on each side
                log_width = (log_freqs[i + 1] - log_freqs[i - 1]) / 2

            # Convert log width to linear bandwidth
            freq_low = 10 ** (log_freqs[i] - log_width / 2)
            freq_high = 10 ** (log_freqs[i] + log_width / 2)
            bandwidths.append(freq_high - freq_low)

        return bandwidths