import numpy as np
from scipy.signal import butter, filtfilt
import struct
from ADPCM import encode_sample, decode_sample
from Subband import SubbandEncoder, SubbandDecoder

class UAC_LC:
    def __init__(self, sample_rate=48000, crossover_freq=4000, frame_size=512):
        self.sample_rate = sample_rate
        self.crossover_freq = crossover_freq
        self.frame_size = frame_size

        # Create low-pass and high-pass filters
        self.lp_b, self.lp_a = self._create_lowpass_filter(crossover_freq)

        # Initialize Subband codec for high frequencies
        self.subband_encoder = SubbandEncoder(
            sample_rate=sample_rate,
            num_bands=17,
            f_min=crossover_freq,
            f_max=16000,
            frame_size=frame_size
        )

        self.subband_decoder = SubbandDecoder(
            sample_rate=sample_rate,
            num_bands=17,
            f_min=crossover_freq,
            f_max=16000,
            frame_size=frame_size
        )

    def _create_lowpass_filter(self, cutoff_freq):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyquist
        return butter(4, normal_cutoff, btype='low')

    def encode(self, signal):
        """Encode audio using hybrid ADPCM (low freq) and Subband (high freq)."""
        # Ensure signal length is multiple of frame_size
        signal = signal[:len(signal) - (len(signal) % self.frame_size)]
        num_frames = len(signal) // self.frame_size

        # Initialize output containers
        adpcm_data = bytearray()
        subband_data = b""

        for i in range(num_frames):
            frame = signal[i * self.frame_size:(i + 1) * self.frame_size]

            # Split frequencies
            low_freq = filtfilt(self.lp_b, self.lp_a, frame)
            high_freq = frame - low_freq  # Get high frequencies by subtraction

            # Encode low frequencies with ADPCM
            low_freq = np.round(low_freq).astype(np.int16)
            for sample in low_freq:
                adpcm_data.append(encode_sample(sample))

            # Encode high frequencies with Subband
            subband_frame = self.subband_encoder.process_frame(high_freq)

            # Pack subband data
            float2int = [int(min(max(value * 32767, -32768), 32767)) for value in subband_frame]
            subband_data += struct.pack(f"{len(float2int)}h", *float2int)

        # Pack everything together
        # Format: [num_frames(4 bytes)][adpcm_length(4 bytes)][adpcm_data][subband_data]
        packed_data = struct.pack("II", num_frames, len(adpcm_data)) + adpcm_data + subband_data
        return packed_data

    def decode(self, packed_data):
        """Decode hybrid encoded audio."""
        # Unpack metadata
        num_frames = struct.unpack("I", packed_data[:4])[0]
        adpcm_length = struct.unpack("I", packed_data[4:8])[0]

        # Split data
        adpcm_data = packed_data[8:8 + adpcm_length]
        subband_data = packed_data[8 + adpcm_length:]

        # Initialize output signal
        reconstructed_signal = np.zeros(num_frames * self.frame_size, dtype=np.float32)

        # Decode ADPCM (low frequencies)
        low_freq_decoded = []
        for byte in adpcm_data:
            decoded_sample = decode_sample(byte)
            low_freq_decoded.append(decoded_sample)

        # Decode Subband (high frequencies)
        subband_frame_size = (self.subband_encoder.num_bands * 2)  # 2 bytes per value
        for frame in range(num_frames):
            start_idx = frame * subband_frame_size
            end_idx = start_idx + subband_frame_size
            frame_data = subband_data[start_idx:end_idx]

            # Unpack the frame data
            subband_values = struct.unpack(f"{self.subband_encoder.num_bands}h", frame_data)
            subband_values = [val / 32767.0 for val in subband_values]

            # Reconstruct high frequencies
            frame_signal = np.zeros(self.frame_size)
            for i, band_value in enumerate(subband_values):
                noise = np.random.randn(self.frame_size) * band_value
                b, a = self.subband_decoder.filters[i]
                filtered_signal = filtfilt(b, a, noise)
                frame_signal += filtered_signal

            # Add to the output signal
            start_sample = frame * self.frame_size
            end_sample = start_sample + self.frame_size
            reconstructed_signal[start_sample:end_sample] += frame_signal

        # Add low and high frequencies
        low_freq_decoded = np.array(low_freq_decoded[:len(reconstructed_signal)])
        reconstructed_signal += low_freq_decoded

        # Normalize and convert to int16
        reconstructed_signal = np.clip(reconstructed_signal * 32768.0, -32768, 32767).astype(np.int16)
        return reconstructed_signal