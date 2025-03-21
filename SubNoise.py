import struct
import numpy as np
import scipy.signal as signal

class SubNoiseEncoder:
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

    def process_frame(self, frame):
        """Process a single frame through the subband encoder."""
        # Normalize input frame to [-1, 1] range
        frame = frame.astype(np.float32) / 32768.0

        # Vectorized processing for all filters
        subband_signals = [
            signal.filtfilt(b, a, frame)
            for b, a in self.filters
        ]
        peak_values = np.max(np.abs(subband_signals), axis=1)
        return peak_values

    def encode(self, signal):
        """Encode the entire signal."""
        # Ensure signal length is multiple of frame_size
        signal = signal[:len(signal) - (len(signal) % self.frame_size)]
        num_frames = len(signal) // self.frame_size

        # Preallocate output array
        all_subband_output = []

        # Process all frames (optimized loop)
        for i in range(num_frames):
            frame = signal[i * self.frame_size:(i + 1) * self.frame_size]
            all_subband_output.append(self.process_frame(frame))

        packed_data = struct.pack("I", num_frames)  # Add number of frames metadata

        for subband_values in all_subband_output:
            if self.resolution == 'int8':
                int8_values = np.clip(subband_values * 127, -128, 127).astype(np.int8)
                packed_data += struct.pack(f"{len(int8_values)}b", *int8_values)
            elif self.resolution == 'int16':
                int16_values = np.clip(subband_values * 32767, -32768, 32767).astype(np.int16)
                packed_data += struct.pack(f"{len(int16_values)}h", *int16_values)
            else:
                raise ValueError("Unsupported resolution. Choose 'int8' or 'int16'.")

        return packed_data

class SubNoiseDecoder:
    def __init__(self, sample_rate=48000, num_bands=32, f_min=4000, f_max=16000, frame_size=512, resolution='int16', noise_type='white'):
        self.fs = sample_rate
        self.num_bands = num_bands - 1  # Adjust for the frequency ranges
        self.frame_size = frame_size
        self.resolution = resolution  # User-defined resolution (e.g., 'int8', 'int16')
        self.noise_type = noise_type  # User-defined noise type ('white', 'pink', etc.)

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

    def _generate_noise(self, num_samples):
        """Generate noise for a specific number of samples."""
        if self.noise_type == 'white':
            return np.random.randn(self.num_bands, num_samples)  # White noise (Gaussian)
        elif self.noise_type == 'pink':
            pink_noise = np.zeros((self.num_bands, num_samples))
            for i in range(self.num_bands):
                pink_noise[i] = np.cumsum(np.random.randn(num_samples))
            return pink_noise
        elif self.noise_type == 'brown':
            brown_noise = np.zeros((self.num_bands, num_samples))
            for i in range(self.num_bands):
                brown_noise[i] = np.cumsum(np.random.randn(num_samples)) * 0.1
            return brown_noise
        else:
            raise ValueError("Unsupported noise type. Choose 'white', 'pink', or 'brown'.")

    def decode(self, packed_data):
        """Decode the packed subband data."""
        # First 4 bytes contain the number of frames
        num_frames = struct.unpack("I", packed_data[:4])[0]
        packed_data = packed_data[4:]  # Remove the metadata

        # Calculate size of each frame's data
        frame_data_size = self.num_bands * (1 if self.resolution == 'int8' else 2)

        reconstructed_signal = []

        # Loop through frames and unpack the data
        for frame in range(num_frames):
            start_idx = frame * frame_data_size
            end_idx = start_idx + frame_data_size
            frame_data = packed_data[start_idx:end_idx]

            # Unpack the frame data
            if self.resolution == 'int8':
                subband_values = np.array(struct.unpack(f"{self.num_bands}b", frame_data), dtype=np.float32)
                subband_values = subband_values / 127.0
            elif self.resolution == 'int16':
                subband_values = np.array(struct.unpack(f"{self.num_bands}h", frame_data), dtype=np.float32)
                subband_values = subband_values / 32767.0
            else:
                raise ValueError("Unsupported resolution. Choose 'int8' or 'int16'.")

            # Generate noise for the current frame
            noise = self._generate_noise(self.frame_size)

            # Apply subband values as scaling factors to the noise
            scaled_noise = noise * subband_values[:, None]

            # Filter each band's noise and sum them
            frame_signal = np.zeros(self.frame_size)
            for i, (b, a) in enumerate(self.filters):
                filtered_signal = signal.filtfilt(b, a, scaled_noise[i])
                frame_signal += filtered_signal

            # Append the processed frame to the output
            reconstructed_signal.extend(frame_signal)

        # Convert the reconstructed signal to int16
        reconstructed_signal = np.clip(np.array(reconstructed_signal) * 32768.0, -32768, 32767).astype(np.int16)
        return reconstructed_signal
