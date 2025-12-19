# # PHSC is Parametric Harmonic Stereo Coding

import numpy as np
from scipy.signal import get_window, find_peaks
from typing import List, Dict, Tuple

class HarmonicStereoExtractor:
    """
    Extracts Parametric Stereo (PS) parameters (IID, ICLD, IPD, ICC) for detected harmonics
    in a stereo audio chunk.
    """

    def __init__(self,
                 sample_rate: int,
                 window_size: int,
                 hop_size: int,
                 min_f0_freq: float,
                 max_f0_freq: float,
                 peak_threshold: float,
                 max_harmonics_per_f0: int,
                 max_harmonic_freq_object: int = 10,
                 log: bool = False):

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.min_f0_freq = min_f0_freq
        self.max_f0_freq = max_f0_freq
        self.peak_threshold = peak_threshold
        self.max_harmonics_per_f0 = max_harmonics_per_f0
        self.max_harmonic_freq_object = max_harmonic_freq_object
        self.log = log

        # Pre-compute window and FFT size
        self.window = get_window('hann', window_size)
        self.fft_size = window_size
        self.freq_resolution = sample_rate / self.fft_size

        self.min_f0_bin = int(self.min_f0_freq / self.freq_resolution)
        self.max_f0_bin = min(int(self.max_f0_freq / self.freq_resolution), self.fft_size // 2)

        if self.log:
            self._precompute_log_bins()

        self._freq_array = np.arange(self.fft_size // 2) * self.freq_resolution

        # State variable for transient detection (using energy of previous frame)
        self.previous_energy_l = 0.0

    def _precompute_log_bins(self):
        """Pre-compute logarithmically spaced frequency bins for log mode."""
        bins_per_octave = 12
        octaves = np.log2(self.max_f0_freq / self.min_f0_freq)
        num_log_bins = int(octaves * bins_per_octave)

        self.log_freqs = np.logspace(
            np.log10(self.min_f0_freq),
            np.log10(self.max_f0_freq),
            num_log_bins
        )

        self.log_bins = np.round(self.log_freqs / self.freq_resolution).astype(int)
        self.log_bins = np.unique(np.clip(self.log_bins, 0, self.fft_size // 2 - 1))

    def process_chunk(self, audio_chunk_l: np.ndarray, audio_chunk_r: np.ndarray) -> List[Dict]:
        """
        Processes a single stereo audio chunk and returns a list of Harmonic Objects.
        """
        if len(audio_chunk_l) != self.window_size:
            # Handle padding if necessary
            def pad_chunk(chunk):
                padded = np.zeros(self.window_size, dtype=chunk.dtype)
                padded[:len(chunk)] = chunk
                return padded

            padded_l = pad_chunk(audio_chunk_l)
            padded_r = pad_chunk(audio_chunk_r)
        else:
            padded_l = audio_chunk_l
            padded_r = audio_chunk_r

        # 1. Apply Windowing
        windowed_l = padded_l * self.window
        windowed_r = padded_r * self.window

        # 2. FFT on both channels
        fft_l = np.fft.rfft(windowed_l, n=self.fft_size)
        fft_r = np.fft.rfft(windowed_r, n=self.fft_size)

        mag_l = np.abs(fft_l)
        phase_l = np.angle(fft_l)
        mag_r = np.abs(fft_r)
        phase_r = np.angle(fft_r)

        max_mag = mag_l.max()

        if max_mag == 0:
            return []

        # --- Transient Detection (Simple Energy Change) ---
        current_energy_l = np.sum(windowed_l ** 2)
        energy_ratio = current_energy_l / (self.previous_energy_l + 1e-6)
        is_transient = (energy_ratio > 4.0) or (energy_ratio < 0.25)
        self.previous_energy_l = current_energy_l

        # --- Multi-Pitch Detection (on Left Channel) ---

        if self.log:
            primary_freq_bins, primary_magnitudes = self._find_peaks_log_scale(
                mag_l, max_mag
            )
        else:
            primary_freq_bins, primary_magnitudes = self._find_peaks_linear(
                mag_l, max_mag
            )

        if len(primary_freq_bins) == 0:
            return []

        # --- Limit to top N strongest peaks ---
        if len(primary_freq_bins) > self.max_harmonic_freq_object:
            top_indices = np.argpartition(primary_magnitudes, -self.max_harmonic_freq_object)[
                          -self.max_harmonic_freq_object:]
            top_indices = top_indices[np.argsort(primary_magnitudes[top_indices])[::-1]]
            primary_freq_bins = primary_freq_bins[top_indices]
            primary_magnitudes = primary_magnitudes[top_indices]

        # --- Harmonic Extraction for Each Primary Peak ---
        harmonic_objects = self._extract_harmonics(
            primary_freq_bins, primary_magnitudes, mag_l, phase_l, mag_r, phase_r, max_mag, is_transient
        )

        return harmonic_objects

    def _find_peaks_linear(self, fft_magnitude: np.ndarray, max_mag: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks using linear frequency scanning."""
        search_start = max(0, self.min_f0_bin)
        search_end = min(len(fft_magnitude), self.max_f0_bin)

        if search_start >= search_end:
            return np.array([]), np.array([])

        peaks, _ = find_peaks(
            fft_magnitude[search_start:search_end],
            height=self.peak_threshold * max_mag
        )

        primary_freq_bins = peaks + search_start
        primary_magnitudes = fft_magnitude[primary_freq_bins]

        return primary_freq_bins, primary_magnitudes

    def _find_peaks_log_scale(self, fft_magnitude: np.ndarray, max_mag: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks using logarithmic frequency scanning."""
        log_magnitudes = fft_magnitude[self.log_bins]

        peaks, _ = find_peaks(
            log_magnitudes,
            height=self.peak_threshold * max_mag
        )

        if len(peaks) == 0:
            return np.array([]), np.array([])

        primary_freq_bins = self.log_bins[peaks]

        refined_bins = []
        refined_mags = []

        for bin_idx in primary_freq_bins:
            if bin_idx > 0 and bin_idx < len(fft_magnitude) - 1:
                alpha = fft_magnitude[bin_idx - 1]
                beta = fft_magnitude[bin_idx]
                gamma = fft_magnitude[bin_idx + 1]

                if beta > alpha and beta > gamma:
                    # Parabolic interpolation for sub-bin accuracy
                    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                    refined_bin = bin_idx + p
                    refined_mag = beta - 0.25 * (alpha - gamma) * p

                    # Round to nearest integer bin for array indexing
                    refined_bins.append(int(np.round(refined_bin)))
                    refined_mags.append(refined_mag)
            else:
                refined_bins.append(bin_idx)
                refined_mags.append(fft_magnitude[bin_idx])

        return np.array(refined_bins), np.array(refined_mags)

    def _calculate_ps_params(self, mag_l: float, phase_l: float, mag_r: float, phase_r: float) -> Dict[str, float]:
        """Calculates Parametric Stereo parameters for a single bin."""
        # 1. Intensity Difference (IID)
        if mag_l > 1e-6:
            iid = 10 * np.log10((mag_r ** 2) / (mag_l ** 2))
        else:
            iid = 30.0

        # 2. Inter-Channel Level Difference (ICLD)
        if mag_l > 1e-6:
            icld = 20 * np.log10(mag_r / mag_l)
        else:
            icld = 15.0

        # 3. Inter-Channel Phase Difference (IPD)
        ipd = phase_r - phase_l
        ipd = np.arctan2(np.sin(ipd), np.cos(ipd))

        # 4. Inter-Channel Coherence/Correlation (ICC)
        # Simple proxy: coherence drops with phase spread
        icc = 1.0 - (np.abs(ipd) / np.pi) * 0.5
        icc = np.clip(icc, 0.0, 1.0)

        return {
            'IID': float(iid),
            'ICLD': float(icld),
            'IPD': float(ipd),
            'ICC': float(icc)
        }

    def _extract_harmonics(self, primary_freq_bins: np.ndarray,
                           primary_magnitudes: np.ndarray,
                           mag_l: np.ndarray,
                           phase_l: np.ndarray,
                           mag_r: np.ndarray,
                           phase_r: np.ndarray,
                           max_mag: float,
                           is_transient: bool) -> List[Dict]:
        """Extract harmonics and PS parameters for all detected primary frequencies."""
        harmonic_objects = []
        threshold = self.peak_threshold * max_mag
        half_fft_size = len(mag_l)
        nyquist_freq = self.sample_rate / 2

        for p0_bin, p0_amp_l in zip(primary_freq_bins, primary_magnitudes):
            if p0_amp_l < threshold:
                continue

            p0_freq = p0_bin * self.freq_resolution
            p0_amp_r = mag_r[p0_bin]
            p0_phase_r = phase_r[p0_bin]

            ps_params = self._calculate_ps_params(p0_amp_l, phase_l[p0_bin], p0_amp_r, p0_phase_r)

            current_harmonics = [{
                'IID': ps_params['IID'],
                'ICLD': ps_params['ICLD'],
                'IPD': ps_params['IPD'],
                'ICC': ps_params['ICC']
            }]

            harmonic_numbers = np.arange(2, self.max_harmonics_per_f0 + 1)
            target_freqs = harmonic_numbers * p0_freq
            valid_mask = (target_freqs <= self.max_f0_freq) & (target_freqs < nyquist_freq)
            harmonic_bins = np.round(harmonic_numbers[valid_mask] * p0_freq / self.freq_resolution).astype(int)
            harmonic_bins = harmonic_bins[harmonic_bins < half_fft_size]

            if len(harmonic_bins) > 0:
                amps_l = mag_l[harmonic_bins]
                phases_l = phase_l[harmonic_bins]
                amps_r = mag_r[harmonic_bins]
                phases_r = phase_r[harmonic_bins]

                for amp_l, phase_l_h, amp_r, phase_r_h in zip(amps_l, phases_l, amps_r, phases_r):
                    ps_params_h = self._calculate_ps_params(amp_l, phase_l_h, amp_r, phase_r_h)

                    current_harmonics.append({
                        'IID': ps_params_h['IID'],
                        'ICLD': ps_params_h['ICLD'],
                        'IPD': ps_params_h['IPD'],
                        'ICC': ps_params_h['ICC']
                    })

            harmonic_objects.append({
                "freq": float(p0_freq),
                "n_harmonic": len(current_harmonics),
                "main_amp_l": float(p0_amp_l),
                "harmonics": current_harmonics
            })

        return harmonic_objects


# --- 2. HARMONIC STEREO SYNTHESIZER CLASS (Copied from phsc.py) ---

class HarmonicStereoSynthesizer:
    """
    Synthesizes stereo audio from a mono signal using Parametric Stereo (PS) parameters
    from harmonic objects.
    """
    def __init__(self, sample_rate: int, window_size: int, hop_size: int, stereo_width: float = 1.0):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = get_window('hann', window_size)
        self.stereo_width = stereo_width  # 1.0 = normal, >1.0 = wider, <1.0 = narrower
    
    def set_stereo_width(self, width: float):
        """
        Set stereo width multiplier.
        width = 1.0: normal stereo
        width > 1.0: expanded stereo (e.g., 1.75 for 175%)
        width < 1.0: narrower stereo
        width = 0.0: mono
        """
        self.stereo_width = np.clip(width, 0.0, 3.0)  # Limit to reasonable range
    
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
        
        # Optional: Apply soft limiting to prevent clipping from extreme widening
        max_val = max(np.max(np.abs(left_out)), np.max(np.abs(right_out)))
        if max_val > 1.0:
            left_out /= max_val
            right_out /= max_val
        
        return left_out, right_out
    
    def _icld_to_lr_gains(self, icld: float) -> Tuple[float, float]:
        """
        Converts ICLD (dB) to normalized L/R amplitude gains assuming constant power.
        """
        # R / L ratio
        ratio_r_l = 10 ** (icld / 20.0)
        # Normalize the gains (constant power: G_L^2 + G_R^2 = 1)
        gain_l = np.sqrt(1.0 / (1 + ratio_r_l ** 2))
        gain_r = gain_l * ratio_r_l
        return gain_l, gain_r
    
    def process_chunk(self, mono_audio_chunk: np.ndarray, harmonic_objects_chunk: List[Dict], prewindow=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Synthesizes stereo audio (L, R) from a mono input signal and PS parameters.
        Now preserves ALL frequencies by starting from the mono FFT and only modifying harmonic bins.
        Applies stereo width expansion as post-processing.
        """
        if not harmonic_objects_chunk:
            empty = np.zeros(self.window_size, dtype=np.float32)
            return empty * self.window, empty * self.window
        
        # FFT of original mono signal
        fft_mono = np.fft.rfft(mono_audio_chunk, n=self.window_size)
        mag_m = np.abs(fft_mono)
        phase_m = np.angle(fft_mono)
        
        # NEW: Start L/R FFTs as *copies* of original mono
        fft_l_out = fft_mono.copy().astype(np.complex64)
        fft_r_out = fft_mono.copy().astype(np.complex64)
        
        # Apply stereo parameters only to harmonic bins
        for h_obj in harmonic_objects_chunk:
            p0_freq = h_obj['freq']
            for n, h in enumerate(h_obj['harmonics'], start=1):
                harmonic_freq = p0_freq * n
                # FFT bin index
                bin_idx = int(np.round(harmonic_freq / self.sample_rate * self.window_size))
                if bin_idx >= len(mag_m):
                    continue
                
                m_amp = mag_m[bin_idx]
                m_phase = phase_m[bin_idx]
                icld = h['ICLD']
                ipd = h['IPD']
                icc = h['ICC']
                
                # L/R gains
                gain_l, gain_r = self._icld_to_lr_gains(icld)
                
                # ICC blend
                ipd_applied = ipd
                if icc < 0.2:
                    avg_gain = (gain_l + gain_r) / 2.0
                    blend_factor = np.clip(icc * 5.0, 0.0, 1.0)
                    gain_l = avg_gain * (1 - blend_factor) + gain_l * blend_factor
                    gain_r = avg_gain * (1 - blend_factor) + gain_r * blend_factor
                    ipd_applied = ipd * blend_factor
                
                # Apply phase offset
                phase_l_out = m_phase - ipd_applied / 2.0
                phase_r_out = m_phase + ipd_applied / 2.0
                
                # Rebuild harmonic bins
                fft_l_out[bin_idx] = (m_amp * gain_l) * (np.cos(phase_l_out) + 1j * np.sin(phase_l_out))
                fft_r_out[bin_idx] = (m_amp * gain_r) * (np.cos(phase_r_out) + 1j * np.sin(phase_r_out))
        
        # Inverse FFT
        output_l = np.fft.irfft(fft_l_out, n=self.window_size)
        output_r = np.fft.irfft(fft_r_out, n=self.window_size)
        
        # POST-PROCESSING: Apply stereo width expansion
        output_l, output_r = self._apply_stereo_width(output_l, output_r)
        
        # Apply window
        if prewindow:
            output_l = output_l * self.window
            output_r = output_r * self.window
        
        return output_l.astype(np.float32), output_r.astype(np.float32)