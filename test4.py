import wave
import numpy as np
from scipy.signal import resample, butter, filtfilt, get_window
from tqdm import tqdm  # Import tqdm for progress bar
import MDCT
from Subband import SubbandEncoder, SubbandDecoder
import Quantizator

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

# Downsampling to specific sample rate for ADPCM encoding
def resample_audio(audio_data, orig_rate, target_rate, chunk_size=1024):
    """
    Resample audio data with respect to the provided chunk size.

    Parameters:
    - audio_data: The audio data array (numpy).
    - orig_rate: The original sample rate.
    - target_rate: The target sample rate to resample to.
    - chunk_size: The size of the chunks to process (default 1024).

    Returns:
    - Resampled audio data (numpy array).
    """
    num_chunks = len(audio_data) // chunk_size
    resampled_data = []

    for i in range(num_chunks):
        chunk = audio_data[i * chunk_size: (i + 1) * chunk_size]
        resampled_chunk = resample(chunk, int(len(chunk) * target_rate / orig_rate))
        resampled_data.append(resampled_chunk)

    # Handle the last chunk if it doesn't fit perfectly into chunks
    remainder = len(audio_data) % chunk_size
    if remainder > 0:
        last_chunk = audio_data[-remainder:]
        resampled_last_chunk = resample(last_chunk, int(len(last_chunk) * target_rate / orig_rate))
        resampled_data.append(resampled_last_chunk)

    # Concatenate all resampled chunks into a single array
    return np.concatenate(resampled_data)

def high_pass_filter(data, cutoff, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Main conversion process
input_wav = './STD_TEST/std_test_input3.wav'
output_wav = "./Output/" + input_wav.split("/")[-1] + '_output.wav'
frame_size = 256
resolution = "int16" # int8 int16
num_bands = 32
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


sbcenc = SubbandEncoder(sample_rate=sample_rate, num_bands=num_bands, frame_size=CHUNK_SIZE // subbandblocks, f_min=mid_max_freq // 2, f_max=subband_max_freq, resolution=resolution)
sbcdec = SubbandDecoder(sample_rate=sample_rate, num_bands=num_bands, frame_size=CHUNK_SIZE // subbandblocks, f_min=mid_max_freq // 2, f_max=subband_max_freq, resolution=resolution)

signal = np.frombuffer(wavfile_input.readframes(total_frames), dtype=np.int16).reshape(-1, 2)

window = get_window('hamming', CHUNK_SIZE)

# Output signal
output_signal = np.zeros_like(signal, dtype=np.int16)

start = 0

# Use tqdm for progress bar
with tqdm(total=total_frames, desc="Processing Audio", unit="chunks") as pbar:
    try:
        while start + CHUNK_SIZE <= len(signal):
            # Read a chunk of audio
            chunk = signal[start:start + CHUNK_SIZE]

            # Apply the window function
            left_channel = chunk[:, 0]
            right_channel = chunk[:, 1]

            # Apply the window to each channel
            left_windowed = left_channel * window
            right_windowed = right_channel * window

            # Stack them back together
            windowed_chunk = np.column_stack((left_windowed, right_windowed))

            # encode process

            # Perform Mid-Side encoding
            mid, side = mid_side_encode(windowed_chunk)

            mid_filtered = high_pass_filter(mid, cutoff=20, sample_rate=sample_rate).clip(-32768, 32768)
            side_filtered = high_pass_filter(side, cutoff=20, sample_rate=sample_rate).clip(-32768, 32768)

            # Downsample mid to 8kHz for ADPCM encoding and side to 4kHz for ADPCM encoding
            mid_downsampled = resample_audio(mid, sample_rate, mid_max_freq, CHUNK_SIZE).astype(np.int16)
            side_downsampled = resample_audio(side, sample_rate, side_max_freq, CHUNK_SIZE).astype(np.int16)

            # Encode mid and side using MDCT
            encoded_mid = MDCT.mdct(mid_downsampled.clip(-32768, 32768))
            encoded_side = MDCT.mdct(side_downsampled.clip(-32768, 32768))

            # Encode side using SBC
            encoded_mid_subband = sbcenc.encode(mid.astype(np.int16))

            print(len(encoded_mid_subband) + len(encoded_mid) + len(encoded_side))

            # decode process

            # Decode mid and side using inverse MDCT (iMDCT)
            decoded_mid = MDCT.imdct(encoded_mid)
            decoded_side = MDCT.imdct(encoded_side)

            decoded_mid_subband = sbcdec.decode(encoded_mid_subband)

            # Upsample decoded mid and decoded side to original sample rate
            decoded_mid_upsampled = resample_audio(decoded_mid, mid_max_freq, sample_rate, CHUNK_SIZE)
            decoded_side_upsampled = resample_audio(decoded_side, side_max_freq, sample_rate, CHUNK_SIZE)

            # Mix decoded_mid_upsampled and decoded_mid_subband
            max_length = max(len(decoded_mid_upsampled), len(decoded_mid_subband))

            # Pad both arrays to the max_length
            decoded_mid_upsampled = np.pad(decoded_mid_upsampled, (0, max(0, max_length - len(decoded_mid_upsampled))),
                                           mode='constant')
            decoded_mid_subband = np.pad(decoded_mid_subband, (0, max(0, max_length - len(decoded_mid_subband))),
                                         mode='constant')

            # Perform the mixing operation
            mixed_mid = (decoded_mid_upsampled) + (decoded_mid_subband * subbandloudness)

            # Ensure that decoded_side_upsampled also has the same length
            decoded_side_upsampled = np.pad(decoded_side_upsampled, (0, max(0, max_length - len(decoded_side_upsampled))), mode='constant')

            # Decode back to left and right channels
            reconstructed_frame = mid_side_decode(mixed_mid.astype(np.int16), decoded_side_upsampled).astype(np.int16)

            # Add the reconstructed frame to the output buffer with clipping
            output_signal[start:start + CHUNK_SIZE] += reconstructed_frame


            # Advance the start position by hop_size
            start += hop_size

            pbar.update(CHUNK_SIZE)


    finally:
        print(output_signal)
        wavfile_output.writeframes(output_signal.astype(np.int16).tobytes())
        wavfile_output.close()