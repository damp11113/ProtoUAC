import numpy as np
import scipy.signal as signal

class SubbandFFTEncoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=20, f_max=16000, window_size=2048):
        self.fs = sample_rate
        self.num_bands = num_bands - 1
        self.window_size = window_size

        # Pre-compute frequency bands
        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max

        # Pre-compute filters
        self.filters = self._precompute_filters(f_min, f_max, num_bands)

        # Pre-compute window
        self.window = np.sin(np.pi * (np.arange(0.5, window_size + 0.5)) / window_size)

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

    def process_signal(self, input_signal):
        """Process the signal through the FFT encoder."""
        input_signal = input_signal.astype(np.float32) / 32768.0

        # Pad signal if needed
        pad_length = (self.window_size - (len(input_signal) % self.window_size)) % self.window_size
        if pad_length:
            input_signal = np.pad(input_signal, (0, pad_length))

        # Reshape into frames
        num_frames = len(input_signal) // self.window_size
        frames = input_signal.reshape(num_frames, self.window_size)

        # Process each subband in parallel using FFT
        subband_signals = []
        for b, a in self.filters:
            filtered = signal.filtfilt(b, a, input_signal)
            filtered_frames = filtered.reshape(num_frames, self.window_size)
            fft_frames = np.fft.rfft(filtered_frames)  # FFT per frame
            subband_signals.append(fft_frames)

        return subband_signals

    def encode(self, input_signal):
        return self.process_signal(input_signal)

class SubbandFFTDecoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=20, f_max=16000, window_size=2048):
        self.fs = sample_rate
        self.num_bands = num_bands - 1
        self.window_size = window_size

        # Pre-compute all constants
        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max
        self.filters = self._precompute_filters(f_min, f_max, num_bands)
        self.window = np.sin(np.pi * (np.arange(0.5, window_size + 0.5)) / window_size)

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

    def decode(self, encoded_data):
        num_frames = len(encoded_data[0])
        M = len(encoded_data[0][0])  # Length of FFT bins
        N = (M - 1) * 2  # Output length after IFFT (2048 from 1025 bins)

        output_length = num_frames * M
        output_signal = np.zeros(output_length + M)

        # Process all subbands in parallel
        for subband_idx, subband_frames in enumerate(encoded_data):
            subband_output = np.zeros_like(output_signal)

            # Process frames
            for i, frame in enumerate(subband_frames):
                # Reconstruct signal from frequency domain using inverse FFT (irfft)
                recovered = np.fft.irfft(frame, N)  # Ensure output length matches original window size
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
