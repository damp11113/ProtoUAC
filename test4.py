import wave
import numpy as np
from scipy.signal import resample, medfilt
from tqdm import tqdm  # Import tqdm for progress bar
from MDCT import mdct, imdct
from Subband import SubbandEncoder, SubbandDecoder

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

def clip_and_smooth(audio_data, threshold=30000):
    # Identify samples that exceed the threshold (likely clipping)
    over_threshold = np.abs(audio_data) > threshold

    # Replace those samples with the average of their neighbors (simple interpolation)
    for idx in np.where(over_threshold)[0]:
        left = audio_data[idx - 1] if idx > 0 else audio_data[idx]
        right = audio_data[idx + 1] if idx < len(audio_data) - 1 else audio_data[idx]
        audio_data[idx] = (left + right) / 2
    return audio_data

def smooth_clipping(audio_data, window_size=5, threshold=30000):
    # Smooth clipping by averaging neighbors around large spikes
    smoothed_data = np.copy(audio_data)
    for i in range(1, len(audio_data) - 1):
        if np.abs(audio_data[i]) > threshold:
            smoothed_data[i] = np.mean(audio_data[max(0, i - window_size): min(len(audio_data), i + window_size)])
    return smoothed_data


def limit_to_db_int(audio_data, target_db=-5):
    """
    Limit the audio data (integer format) to a specific dB level.

    Parameters:
    - audio_data: The audio data to limit (numpy array of integers).
    - target_db: The target dB level to limit to (default is -6 dB).

    Returns:
    - The limited audio data (numpy array of integers).
    """
    # Calculate the current peak level of the audio (as integer max absolute value)
    peak = np.max(np.abs(audio_data))

    # The maximum possible value for 16-bit signed PCM is 32767 (for int16)
    max_int16_value = 32767

    # Calculate the scaling factor to reach the target dB
    target_amplitude = 10 ** (target_db / 20)

    # Calculate the scaling factor to limit the peak to the target level
    scale_factor = target_amplitude * max_int16_value / peak

    # Apply the scaling factor and convert back to int16 (clipping if necessary)
    limited_audio = np.clip(audio_data * scale_factor, -max_int16_value, max_int16_value).astype(np.int16)

    return limited_audio


# Main conversion process
input_wav = './STD_TEST/std_test_input.wav'
output_wav = "./Output/" + input_wav.split("/")[-1] + '_output.wav'
frame_size = 256
resolution = "int8" # int8 int16
num_bands = 32

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

num_chunks = total_frames // CHUNK_SIZE

sbcenc = SubbandEncoder(sample_rate=sample_rate, num_bands=num_bands, frame_size=CHUNK_SIZE // 8, f_min=4000, f_max=16000, resolution=resolution)
sbcdec = SubbandDecoder(sample_rate=sample_rate, num_bands=num_bands, frame_size=CHUNK_SIZE // 8, f_min=4000, f_max=16000, resolution=resolution)

# Use tqdm for progress bar
for _ in tqdm(range(num_chunks), desc="Processing audio chunks", unit="chunk"):
    # Read a chunk of audio
    frame_data = wavfile_input.readframes(CHUNK_SIZE)

    # Convert to numpy array and extract left and right channels
    frame_block = np.frombuffer(frame_data, dtype=np.int16).reshape(-1, 2)

    frame_block = limit_to_db_int(frame_block, target_db=-5)

    # encode process

    # Perform Mid-Side encoding
    mid, side = mid_side_encode(frame_block)

    # Downsample mid to 8kHz for ADPCM encoding and side to 4kHz for ADPCM encoding
    mid_downsampled = resample_audio(mid, sample_rate, 8000, CHUNK_SIZE).astype(np.int16)
    side_downsampled = resample_audio(side, sample_rate, 4000, CHUNK_SIZE).astype(np.int16)

    # Encode mid and side using MDCT
    encoded_mid = mdct(mid_downsampled)
    encoded_side = mdct(side_downsampled)

    # Encode side using SBC
    encoded_mid_subband = sbcenc.encode(mid.astype(np.int16))

    # packaging process


    print(len(encoded_mid_subband) + len(encoded_mid) + len(encoded_side))

    # decode process

    # Decode mid and side using inverse MDCT (iMDCT)
    decoded_mid = imdct(encoded_mid)
    decoded_side = imdct(encoded_side)

    decoded_mid_subband = clip_and_smooth(sbcdec.decode(encoded_mid_subband))

    # Upsample decoded mid and decoded side to original sample rate
    decoded_mid_upsampled = resample_audio(decoded_mid, 8000, sample_rate, CHUNK_SIZE)
    decoded_side_upsampled = resample_audio(decoded_side, 4000, sample_rate, CHUNK_SIZE)

    # Mix decoded_mid_upsampled and decoded_mid_subband
    max_length = max(len(decoded_mid_upsampled), len(decoded_mid_subband))
    decoded_mid_upsampled = np.pad(decoded_mid_upsampled, (0, max_length - len(decoded_mid_upsampled)), mode='constant')
    decoded_mid_subband = np.pad(decoded_mid_subband, (0, max_length - len(decoded_mid_subband)), mode='constant')
    mixed_mid = decoded_mid_upsampled + (decoded_mid_subband * 3.5).astype(np.int16)

    # Ensure that decoded_side_upsampled also has the same length
    decoded_side_upsampled = np.pad(decoded_side_upsampled, (0, max_length - len(decoded_side_upsampled)), mode='constant')

    # Decode back to left and right channels
    reconstructed_frame = mid_side_decode(mixed_mid, decoded_side_upsampled).astype(np.int16)

    # Normalize and write the output to the fil
    output = reconstructed_frame
    wavfile_output.writeframes(output.tobytes())

