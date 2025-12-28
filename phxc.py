# PHXC is Parametric Harmonic eXtraction Coding

import numpy as np
from scipy.signal import get_window, find_peaks
from typing import List, Dict

class HarmonicExtractor:
    def __init__(self, sample_rate, window_size, hop_size, min_f0_freq, max_f0_freq, 
                 peak_threshold, max_harmonics_per_f0, max_harmonic_freq_output, 
                 max_harmonic_freq_object=10, log=False):
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

        self.window = get_window('hann', window_size).astype(np.float32)
        self.freq_resolution = sample_rate / window_size
        self.inv_freq_res = 1.0 / self.freq_resolution
        
        self.min_f0_bin = int(min_f0_freq * self.inv_freq_res)
        self.max_f0_bin = min(int(max_f0_freq * self.inv_freq_res), window_size // 2)

        if self.log:
            self._precompute_log_bins()

    def _precompute_log_bins(self):
        bins_per_octave = 12
        octaves = np.log2(self.max_f0_freq / self.min_f0_freq)
        num_log_bins = int(octaves * bins_per_octave)
        log_freqs = np.logspace(np.log10(self.min_f0_freq), np.log10(self.max_f0_freq), num_log_bins)
        self.log_bins = np.unique(np.clip(np.round(log_freqs * self.inv_freq_res).astype(int), 0, self.window_size // 2 - 1))

    def process_chunk(self, audio_chunk: np.ndarray) -> List[Dict]:
        if len(audio_chunk) < self.window_size:
            padded = np.zeros(self.window_size, dtype=np.float32)
            padded[:len(audio_chunk)] = audio_chunk
            audio_chunk = padded

        fft_res = np.fft.rfft(audio_chunk * self.window)
        mag = np.abs(fft_res)
        max_m = mag.max()
        if max_m < 1e-9: return []

        if self.log:
            bins, amps = self._find_peaks_log_scale(mag, max_m)
        else:
            bins, amps = self._find_peaks_linear(mag, max_m)

        if len(bins) == 0: return []

        if len(bins) > self.max_harmonic_freq_object:
            idx = np.argpartition(amps, -self.max_harmonic_freq_object)[-self.max_harmonic_freq_object:]
            idx = idx[np.argsort(amps[idx])[::-1]]
            bins, amps = bins[idx], amps[idx]

        return self._extract_harmonics(bins, amps, mag, np.angle(fft_res), max_m)

    def _find_peaks_linear(self, mag, max_m):
        start, end = self.min_f0_bin, self.max_f0_bin
        p, _ = find_peaks(mag[start:end], height=self.peak_threshold * max_m)
        return p + start, mag[p + start]

    def _find_peaks_log_scale(self, mag, max_m):
        l_mag = mag[self.log_bins]
        p, _ = find_peaks(l_mag, height=self.peak_threshold * max_m)
        if len(p) == 0: return np.array([]), np.array([])
        
        orig_bins = self.log_bins[p]
        refined_bins, refined_mags = [], []
        for b in orig_bins:
            if 0 < b < len(mag) - 1:
                a, b_val, g = mag[b-1], mag[b], mag[b+1]
                den = (a - 2*b_val + g)
                p_shift = 0.5 * (a - g) / den if den != 0 else 0
                refined_bins.append(int(round(b + p_shift)))
                refined_mags.append(b_val - 0.25 * (a - g) * p_shift)
            else:
                refined_bins.append(b); refined_mags.append(mag[b])
        return np.array(refined_bins), np.array(refined_mags)

    def _extract_harmonics(self, bins, amps, mag, phase, max_m):
        objs = []
        thresh = self.peak_threshold * max_m
        limit = min(self.max_harmonic_freq_output, (self.sample_rate / 2) - 1)
        
        for b, a in zip(bins, amps):
            if a < thresh: continue
            f0 = b * self.freq_resolution
            
            if f0 <= 0: continue
            
            h_count = min(self.max_harmonics_per_f0, int(limit / f0))
            
            h_nums = np.arange(1, h_count + 1)
            h_bins = np.round(h_nums * f0 * self.inv_freq_res).astype(int)
            valid = h_bins < len(mag)
            h_bins = h_bins[valid]
            
            # Convert FFT magnitude to approximate time-domain amplitude
            # For a real sinusoid, amplitude ~= 2*|FFT_bin|/N (rough approximation)
            scale = 2.0 / float(self.window_size)
            h_list = [
                {'amp': float(mag[hb] * scale), 'phase': float(phase[hb])}
                for hb in h_bins
            ]

            objs.append({
                "freq": float(f0),
                "n_harmonic": len(h_list),
                "main_amp": float(a * scale),
                "harmonics": h_list
            })
        return objs

class HarmonicGenerator:
    def __init__(self, sample_rate: int, window_size: int, hop_size: int):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = get_window('hann', window_size).astype(np.float32)
        self.t = np.arange(window_size, dtype=np.float32) / sample_rate
        self.two_pi = 2.0 * np.pi

    def process_chunk(self, chunk_data: list) -> np.ndarray:
        if not chunk_data: return np.zeros(self.window_size, dtype=np.float32)

        total_signal = np.zeros(self.window_size, dtype=np.float32)
        for obj in chunk_data:
            f0 = obj['freq']
            for i, h in enumerate(obj['harmonics'], 1):
                freq = f0 * i
                phase_inc = self.two_pi * freq * self.t + h['phase']
                total_signal += h['amp'] * np.sin(phase_inc)
        
        return total_signal.astype(np.float32)