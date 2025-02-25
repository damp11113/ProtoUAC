import numpy as np
import scipy.signal as signal

class SubbandMDCTEncoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=20, f_max=16000, window_size=2048):
        self.fs = sample_rate
        self.num_bands = num_bands - 1
        self.window_size = window_size

        # Pre-compute frequency bands
        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max

        # Pre-compute filters
        self.filters = self._precompute_filters(f_min, f_max, num_bands)

        # Pre-compute window and MDCT constants
        self.window = np.sin(np.pi * (np.arange(0.5, window_size + 0.5)) / window_size)
        self._precompute_mdct_constants()

    def _precompute_filters(self, f_min, f_max, num_bands):
        filters = []
        for i in range(num_bands - 1):
            lowcut = self.bands_freq[i]
            highcut = self.bands_freq[i + 1]
            nyquist = 0.5 * self.fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filters.append((b, a))
        return filters

    def _precompute_mdct_constants(self):
        N = self.window_size
        n0 = (N // 2 + 1) // 2
        M = N // 2
        k = np.arange(M)
        n = np.arange(N)
        self.phase_matrix = np.cos(np.pi / M * (n[:, None] + n0) * (k[None, :] + 0.5))
        self.scale_factor = np.sqrt(2 / M)

    @staticmethod
    def _fast_mdct(x, window, phase_matrix, scale_factor):
        """Optimized MDCT computation"""
        windowed = x * window
        return scale_factor * (windowed @ phase_matrix)

    def process_signal(self, input_signal):
        """Process the signal through the MDCT encoder."""
        input_signal = input_signal.astype(np.float32) / 32768.0

        # Pad signal if needed
        pad_length = (self.window_size - (len(input_signal) % self.window_size)) % self.window_size
        if pad_length:
            input_signal = np.pad(input_signal, (0, pad_length))

        # Reshape into frames
        num_frames = len(input_signal) // self.window_size
        frames = input_signal.reshape(num_frames, self.window_size)

        # Process each subband in parallel using vectorized operations
        subband_signals = []
        for b, a in self.filters:
            filtered = signal.filtfilt(b, a, input_signal)
            filtered_frames = filtered.reshape(num_frames, self.window_size)
            mdct_frames = np.array([self._fast_mdct(frame, self.window, self.phase_matrix, self.scale_factor)
                                    for frame in filtered_frames])
            subband_signals.append(mdct_frames)

        return subband_signals

    def encode(self, input_signal):
        return self.process_signal(input_signal)


class SubbandMDCTDecoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=20, f_max=16000, window_size=2048):
        self.fs = sample_rate
        self.num_bands = num_bands - 1
        self.window_size = window_size

        # Pre-compute all constants
        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max
        self.filters = self._precompute_filters(f_min, f_max, num_bands)
        self.window = np.sin(np.pi * (np.arange(0.5, window_size + 0.5)) / window_size)
        self._precompute_imdct_constants()

    def _precompute_filters(self, f_min, f_max, num_bands):
        filters = []
        for i in range(num_bands - 1):
            lowcut = self.bands_freq[i]
            highcut = self.bands_freq[i + 1]
            nyquist = 0.5 * self.fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filters.append((b, a))
        return filters

    def _precompute_imdct_constants(self):
        M = self.window_size // 2
        N = M * 2
        n0 = (N // 2 + 1) // 2
        k = np.arange(M)
        n = np.arange(N)
        self.imdct_matrix = np.cos(np.pi / M * (n[:, None] + n0) * (k[None, :] + 0.5))
        self.scale_factor = np.sqrt(2 / M)

    @staticmethod
    def _fast_imdct(X, window, imdct_matrix, scale_factor):
        """Optimized IMDCT computation"""
        x = scale_factor * (X @ imdct_matrix.T)
        return x * window

    def decode(self, encoded_data):
        num_frames = len(encoded_data[0])
        M = len(encoded_data[0][0])
        N = M * 2

        output_length = num_frames * M
        output_signal = np.zeros(output_length + M)

        # Process all subbands in parallel
        for subband_idx, subband_frames in enumerate(encoded_data):
            subband_output = np.zeros_like(output_signal)

            # Process frames
            for i, frame in enumerate(subband_frames):
                recovered = self._fast_imdct(frame, self.window, self.imdct_matrix, self.scale_factor)
                start_idx = i * M
                subband_output[start_idx:start_idx + N] += recovered

            # Apply bandpass filtering
            filtered_output = signal.filtfilt(self.filters[subband_idx][0],
                                              self.filters[subband_idx][1],
                                              subband_output)
            output_signal += filtered_output

        # Trim and convert to int16
        output_signal = output_signal[:output_length]
        return np.clip(output_signal * 32768.0, -32768, 32767).astype(np.int16)