import wave
import numpy as np
from scipy.signal import resample, butter, filtfilt, get_window
from tqdm import tqdm  # Import tqdm for progress bar
import MDCT
from SubbandDCT import SubbandDCTEncoder, SubbandDCTDecoder
from SubbandMDCT import SubbandMDCTEncoder, SubbandMDCTDecoder
from SubbandFFT import SubbandFFTEncoder, SubbandFFTDecoder
from SubNoise import SubNoiseEncoder, SubNoiseDecoder
import Quantizator
import ADPCM


# Mid-Side Encoding (Stereo to Mid-Side conversion)
def mid_side_encode(stereo_data):
    # Assuming stereo_data is a numpy array of shape (N, 2), where N is the number of frames
    left_channel = stereo_data[:, 0]
    right_channel = stereo_data[:, 1]

    # Mid is the sum of left and right channels
    mid = (left_channel + right_channel) / 2

    # Side is the difference of left and right channels
    side = (left_channel - right_channel) / 2

    return mid, side

def mid_side_decode(mid, side):
    # To decode, you reconstruct the left and right channels
    left_channel = mid + side
    right_channel = mid - side
    return np.column_stack((left_channel, right_channel))

# Main conversion process
input_wav = './STD_TEST/DumDum_compressed.wav'
output_wav = "./Output/" + input_wav.split("/")[-1] + '4_output.wav'
frame_size = 256
resolution = "int16" # int8 int16
num_bands = 32
num_bands_Subband = 32
mid_max_freq = 8000
side_max_freq = 2000
subband_max_freq = 16000
subbandloudness = 5
subbandblocks = 8

wavfile_input = wave.open(input_wav, 'rb')
wavfile_output = wave.open(output_wav, 'wb')

sample_rate = wavfile_input.getframerate()
sample_width = wavfile_input.getsampwidth()

wavfile_output.setnchannels(2)
wavfile_output.setsampwidth(sample_width)
wavfile_output.setframerate(sample_rate)

# Get total number of frames for progress bar
total_frames = wavfile_input.getnframes()

CHUNK_SIZE = int((frame_size / 1000) * sample_rate)
print(CHUNK_SIZE)

hop_size = CHUNK_SIZE // 2  # Overlap between chunks (e.g., 50% overlap)

Mencoder = SubbandFFTEncoder(sample_rate=sample_rate, num_bands=num_bands_Subband, f_min=20, f_max=mid_max_freq)
Sencoder = SubbandFFTEncoder(sample_rate=sample_rate, num_bands=num_bands_Subband, f_min=20, f_max=mid_max_freq)

Mdecoder = SubbandFFTDecoder(sample_rate=sample_rate, num_bands=num_bands_Subband, f_min=20, f_max=mid_max_freq)
Sdecoder = SubbandFFTDecoder(sample_rate=sample_rate, num_bands=num_bands_Subband, f_min=20, f_max=mid_max_freq)

signal = np.frombuffer(wavfile_input.readframes(total_frames), dtype=np.int16).reshape(-1, 2)

window = get_window('hamming', CHUNK_SIZE)

# Output signal
output_signal = np.zeros_like(signal, dtype=np.int16)

start = 0

# Use tqdm for progress bar
with tqdm(total=total_frames, desc="Processing Audio", unit="chunks") as pbar:
    # Debugging and ensuring consistent frame size

    print(f"CHUNK_SIZE: {CHUNK_SIZE}")  # Debug to check the value of CHUNK_SIZE

    # Inside your main processing loop
    try:
        while start + CHUNK_SIZE <= len(signal):
            chunk = signal[start:start + CHUNK_SIZE]

            # Apply windowing
            left_windowed = chunk[:, 0] * window
            right_windowed = chunk[:, 1] * window
            windowed_chunk = np.column_stack((left_windowed, right_windowed))

            # Encode
            mid, side = mid_side_encode(windowed_chunk)

            encoded_mid = Mencoder.encode(mid)
            encoded_side = Sencoder.encode(side)

            # Decode
            decoded_mid = Mdecoder.decode(encoded_mid)
            decoded_side = Sdecoder.decode(encoded_side)

            # Ensure decoded outputs match CHUNK_SIZE
            # Resize if necessary (truncate or pad)
            decoded_mid_resized = decoded_mid[:CHUNK_SIZE]
            decoded_side_resized = decoded_side[:CHUNK_SIZE]

            # Pad or truncate if necessary to match CHUNK_SIZE
            if decoded_mid_resized.shape[0] < CHUNK_SIZE:
                padding = CHUNK_SIZE - decoded_mid_resized.shape[0]
                decoded_mid_resized = np.pad(decoded_mid_resized, ((0, padding), (0, 0)), mode='constant')
                decoded_side_resized = np.pad(decoded_side_resized, ((0, padding), (0, 0)), mode='constant')

            # Debug: Ensure the arrays are the right shape before mixing
            print(f"Decoded mid shape: {decoded_mid_resized.shape}")
            print(f"Decoded side shape: {decoded_side_resized.shape}")

            # Mix and reconstruct
            reconstructed_frame = mid_side_decode(decoded_mid_resized, decoded_side_resized)

            # Ensure the reconstructed frame is of the correct shape (CHUNK_SIZE, 2)
            reconstructed_frame = reconstructed_frame[:CHUNK_SIZE, :]

            # Debug: Ensure the reconstructed frame has the right shape
            print(f"Reconstructed frame shape: {reconstructed_frame.shape}")

            # Use overlap-add for smooth transitions
            output_signal[start:start + CHUNK_SIZE] = reconstructed_frame

            start += hop_size
            pbar.update(hop_size)

    finally:
        print(output_signal)
        wavfile_output.writeframes(output_signal.astype(np.int16).tobytes())
        wavfile_output.close()


