import numpy as np
import scipy.signal as signal
from scipy.fft import dct, idct

class SubbandDCTEncoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=4000, f_max=16000):
        self.fs = sample_rate
        self.num_bands = num_bands - 1  # Adjust for the frequency ranges

        # Create frequency bands
        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max

        self.bands_freq_range = [f"{self.bands_freq[i]}-{self.bands_freq[i + 1]}" for i in
                                 range(len(self.bands_freq) - 1)]

        # Create bandpass filters
        self.filters = [self._create_bandpass_filter(*map(int, band.split('-'))) for band in self.bands_freq_range]

    def _create_bandpass_filter(self, lowcut, highcut):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return b, a

    def _dct_encode(self, input_signal):
        return dct(input_signal, type=2, norm='ortho')

    def process_signal(self, input_signal):
        """Process the entire signal through the DCT encoder."""
        input_signal = input_signal.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        subband_signals = [
            signal.filtfilt(b, a, input_signal)  # Filter the signal through each bandpass filter
            for b, a in self.filters
        ]
        # Apply DCT to each subband signal
        encoded_subbands = [self._dct_encode(subband) for subband in subband_signals]
        return encoded_subbands

    def encode(self, input_signal):
        """Encode the entire signal."""
        return self.process_signal(input_signal)


class SubbandDCTDecoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=4000, f_max=16000):
        self.fs = sample_rate
        self.num_bands = num_bands - 1

        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max
        self.bands_freq_range = [f"{self.bands_freq[i]}-{self.bands_freq[i + 1]}" for i in
                                 range(len(self.bands_freq) - 1)]

        self.filters = [self._create_bandpass_filter(*map(int, band.split('-'))) for band in self.bands_freq_range]

    def _create_bandpass_filter(self, lowcut, highcut):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return b, a

    def _idct_decode(self, encoded_signal):
        # Ensure the signal is 1D before applying IDCT
        encoded_signal = np.reshape(encoded_signal, -1)
        return idct(encoded_signal, type=2, norm='ortho')

    def decode(self, encoded_data):
        """Decode the entire signal."""
        reconstructed_signal = np.zeros(len(encoded_data[0]))  # Assuming all subbands are of equal length

        for i, subband_values in enumerate(encoded_data):
            # Ensure subband_values is 1D before applying IDCT
            subband_values = np.reshape(subband_values, -1)

            # Apply IDCT to each subband signal
            decoded_subband = self._idct_decode(subband_values)

            # Reconstruct the signal by summing the decoded subbands
            filtered_signal = signal.filtfilt(self.filters[i][0], self.filters[i][1], decoded_subband)
            reconstructed_signal += filtered_signal

        # Convert the reconstructed signal to int16
        reconstructed_signal = np.clip(np.array(reconstructed_signal) * 32768.0, -32768, 32767).astype(np.int16)
        return reconstructed_signal
