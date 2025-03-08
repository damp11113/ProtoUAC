import struct
import numpy as np
import scipy.signal as signal

class SubPSEncoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=4000, f_max=16000, frame_size=512, resolution='int16'):
        self.fs = sample_rate
        self.num_bands = num_bands - 1  # Adjust for the frequency ranges
        self.frame_size = frame_size
        self.resolution = resolution  # User-defined resolution (e.g., 'int8', 'int16')

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
        # Reduce the filter order for higher frequencies
        b, a = signal.butter(4, [low, high], btype='band')
        return b, a

    def _calculate_icld_vectorized(self, subbandsL, subbandsR):
        power_left = np.sqrt(np.mean(np.square(subbandsL), axis=1))
        power_right = np.sqrt(np.mean(np.square(subbandsR), axis=1))

        icld = 20 * np.log10(power_left / power_right)

        # Ensure that positive values indicate left dominance, and negative indicate right
        icld = np.where(power_left >= power_right, np.abs(icld), -np.abs(icld))

        return icld

    def process_frame(self, frame):
        """Process a single frame through the subband encoder."""
        # Normalize input frame to [-1, 1] range
        frame = frame.astype(np.float32) / 32768.0

        # Split channels
        left_channel = frame[:, 0]
        right_channel = frame[:, 1]

        # Vectorized processing for all filters
        subband_signals_left = [signal.filtfilt(b, a, left_channel) for b, a in self.filters]
        subband_signals_right = [signal.filtfilt(b, a, right_channel) for b, a in self.filters]

        return self._calculate_icld_vectorized(subband_signals_left, subband_signals_right)

    def encode(self, signal):
        """Encode the entire signal."""
        # Ensure signal length is multiple of frame_size
        signal = signal[:len(signal) - (len(signal) % self.frame_size)]
        num_frames = len(signal) // self.frame_size

        # Preallocate output array
        mono_output = []
        para_output = []

        # Process all frames (optimized loop)
        for i in range(num_frames):
            frame = signal[i * self.frame_size:(i + 1) * self.frame_size]

            left_frame = frame[:, 0]  # Changed indexing to handle 2D array
            right_frame = frame[:, 1]

            mono_frame = (left_frame + right_frame) / 2.0
            mono_output.extend(mono_frame)

            para_output.append(self.process_frame(frame))

        mono_output = np.array(mono_output).astype(np.int16)
        packed_data = struct.pack("I", num_frames)

        for subband_values in para_output:
            if self.resolution == 'int8':
                int8_values = np.clip(subband_values * 127, -128, 127).astype(np.int8)
                packed_data += struct.pack(f"{len(int8_values)}b", *int8_values)
            elif self.resolution == 'int16':
                int16_values = np.clip(subband_values * 32767, -32768, 32767).astype(np.int16)
                packed_data += struct.pack(f"{len(int16_values)}h", *int16_values)
            else:
                raise ValueError("Unsupported resolution. Choose 'int8' or 'int16'.")

        return mono_output, packed_data

class SubPSDecoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=4000, f_max=16000, frame_size=512, resolution='int16', smoothing_factor=0.1):
        self.fs = sample_rate
        self.num_bands = num_bands - 1  # Adjust for the frequency ranges
        self.frame_size = frame_size
        self.resolution = resolution  # User-defined resolution (e.g., 'int8', 'int16')

        # Smoothing factor for panning
        self.smoothing_factor = smoothing_factor
        self.prev_gain_left = 1.0
        self.prev_gain_right = 1.0

        # Create frequency bands
        self.bands_freq = np.round(np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)).astype(int)
        self.bands_freq[-1] = f_max

        self.bands_freq_range = [f"{self.bands_freq[i]}-{self.bands_freq[i + 1]}" for i in range(len(self.bands_freq) - 1)]

        # Create bandpass filters for reconstruction
        self.filters = [self._create_bandpass_filter(*map(int, band.split('-'))) for band in self.bands_freq_range]

        # Create low-pass and high-pass filters for out-of-range frequencies
        self.lowpass_b, self.lowpass_a = self._create_lowpass_filter(f_min)
        self.highpass_b, self.highpass_a = self._create_highpass_filter(f_max)

    def _create_bandpass_filter(self, lowcut, highcut):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return b, a

    def _create_lowpass_filter(self, cutoff):
        nyquist = 0.5 * self.fs
        low = cutoff / nyquist
        b, a = signal.butter(4, low, btype='low')
        return b, a

    def _create_highpass_filter(self, cutoff):
        nyquist = 0.5 * self.fs
        high = cutoff / nyquist
        b, a = signal.butter(4, high, btype='high')
        return b, a

    def icld_apply(self, mono_signal, icld_value):
        # Convert ICLD (dB) to linear scale
        gain_left = 10 ** (icld_value / 20.0)
        gain_right = 10 ** (-icld_value / 20.0)

        # Smooth the gain transitions over time
        gain_left = self.smooth_gain(self.prev_gain_left, gain_left)
        gain_right = self.smooth_gain(self.prev_gain_right, gain_right)

        # Normalize gains to maintain energy balance
        norm_factor = np.sqrt(1 / (gain_left ** 2 + gain_right ** 2))
        gain_left *= norm_factor
        gain_right *= norm_factor

        # Store the current gains for the next frame
        self.prev_gain_left = gain_left
        self.prev_gain_right = gain_right

        # Apply panning
        left = mono_signal * gain_left
        right = mono_signal * gain_right

        return left, right

    def smooth_gain(self, prev_gain, current_gain):
        # Apply exponential smoothing to the gain values
        return (1 - self.smoothing_factor) * prev_gain + self.smoothing_factor * current_gain

    def decode(self, mono_signal, packed_data):
        num_frames = struct.unpack("I", packed_data[:4])[0]
        packed_data = packed_data[4:]  # Remove metadata

        frame_data_size = self.num_bands * (1 if self.resolution == 'int8' else 2)
        reconstructed_signal_left = []
        reconstructed_signal_right = []

        for frame in range(num_frames):
            start_idx = frame * frame_data_size
            end_idx = start_idx + frame_data_size
            frame_data = packed_data[start_idx:end_idx]

            if self.resolution == 'int8':
                subband_values = np.array(struct.unpack(f"{self.num_bands}b", frame_data), dtype=np.float32)
            elif self.resolution == 'int16':
                subband_values = np.array(struct.unpack(f"{self.num_bands}h", frame_data), dtype=np.float32)
            else:
                raise ValueError("Unsupported resolution. Choose 'int8' or 'int16'.")

            left_frame = np.zeros(self.frame_size)
            right_frame = np.zeros(self.frame_size)

            low_freq = signal.filtfilt(self.lowpass_b, self.lowpass_a,
                                         mono_signal[frame * self.frame_size:(frame + 1) * self.frame_size])
            high_freq = signal.filtfilt(self.highpass_b, self.highpass_a,
                                          mono_signal[frame * self.frame_size:(frame + 1) * self.frame_size])

            for band_idx, (b, a) in enumerate(self.filters):
                subband = signal.filtfilt(b, a, mono_signal[frame * self.frame_size:(frame + 1) * self.frame_size])
                gain_left, gain_right = self.icld_apply(subband, subband_values[band_idx])
                left_frame += gain_left
                right_frame += gain_right

            left_frame += low_freq + high_freq
            right_frame += low_freq + high_freq

            reconstructed_signal_left.extend(left_frame)
            reconstructed_signal_right.extend(right_frame)

        reconstructed_signal_left = np.clip(np.array(reconstructed_signal_left), -32768, 32767).astype(
            np.int16)
        reconstructed_signal_right = np.clip(np.array(reconstructed_signal_right), -32768, 32767).astype(
            np.int16)

        stereo_signal = np.zeros((len(reconstructed_signal_left) * 2,), dtype=np.int16)
        stereo_signal[0::2] = reconstructed_signal_left
        stereo_signal[1::2] = reconstructed_signal_right

        return stereo_signal.reshape((-1, 2))
