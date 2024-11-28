import wave
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm  # Import tqdm for progress bar
import ADPCM
from Subband import SubbandEncoder, SubbandDecoder

# Normalize function to prevent clipping and loudness issues
def normalize_audio(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = (audio_data / max_val) * 32767  # Normalize to 16-bit signed integer range
    return np.round(audio_data).astype(np.int16)

# High-pass filter to remove DC offset and low-frequency noise (optional)
def highpass_filter(audio_data, sample_rate=8000, cutoff_freq=20):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, audio_data)
    return filtered_data

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
def resample_audio(audio_data, orig_rate, target_rate):
    return resample(audio_data, int(len(audio_data) * target_rate / orig_rate))


# Main conversion process
input_wav = 'input.wav'
output_wav = 'output.wav'
CHUNK_SIZE = 1024

wavfile_input = wave.open(input_wav, 'rb')
wavfile_output = wave.open(output_wav, 'wb')

sample_rate = wavfile_input.getframerate()
sample_width = wavfile_input.getsampwidth()

wavfile_output.setnchannels(2)
wavfile_output.setsampwidth(sample_width)
wavfile_output.setframerate(sample_rate)

# Get total number of frames for progress bar
total_frames = wavfile_input.getnframes()
num_chunks = total_frames // CHUNK_SIZE

sbcenc = SubbandEncoder(sample_rate=sample_rate, num_bands=32, frame_size=CHUNK_SIZE // 2, f_min=4000, f_max=16000)
sbcdec = SubbandDecoder(sample_rate=sample_rate, num_bands=32, frame_size=CHUNK_SIZE // 2, f_min=4000, f_max=16000)

# Use tqdm for progress bar
for _ in tqdm(range(num_chunks), desc="Processing audio chunks", unit="chunk"):
    # Read a chunk of audio
    frame_data = wavfile_input.readframes(CHUNK_SIZE)

    # Convert to numpy array and extract left and right channels
    frame_block = np.frombuffer(frame_data, dtype=np.int16).reshape(-1, 2)

    # Perform Mid-Side encoding
    mid, side = mid_side_encode(frame_block)

    # Downsample mid to 8kHz for ADPCM encoding and side to 4kHz for ADPCM encoding
    mid_downsampled = resample_audio(mid, sample_rate, 8000).astype(np.int16)
    side_downsampled = resample_audio(side, sample_rate, 4000).astype(np.int16)

    # ADPCM encode mid and side channels (with their respective sample rates)
    encoded_mid = np.array([ADPCM.encode_sample(sample) for sample in mid_downsampled])
    encoded_side = np.array([ADPCM.encode_sample(sample) for sample in side_downsampled])

    # Encode side using SBC
    encoded_mid_subband = sbcenc.encode(mid.astype(np.int16))


    # Reconstruct the audio signal
    decoded_mid = np.array([ADPCM.decode_sample(neeble) for neeble in encoded_mid])
    decoded_side = np.array([ADPCM.decode_sample(neeble) for neeble in encoded_side])

    decoded_mid_subband = sbcdec.decode(encoded_mid_subband)

    # Upsample decoded mid and decoded side to original sample rate
    decoded_mid_upsampled = resample_audio(decoded_mid, 8000, sample_rate)
    decoded_side_upsampled = resample_audio(decoded_side, 4000, sample_rate)

    # Mix decoded_mid_upsampled and decoded_side_subband
    min_length = min(len(decoded_mid_upsampled), len(decoded_mid_subband))
    decoded_mid_upsampled = decoded_mid_upsampled[:min_length]
    decoded_mid_subband = decoded_mid_subband[:min_length]
    mixed_mid = decoded_mid_upsampled + decoded_mid_subband

    # Decode back to left and right channels
    reconstructed_frame = mid_side_decode(mixed_mid, decoded_side_upsampled)

    # Normalize and write the output to the file
    output = normalize_audio(reconstructed_frame.flatten())
    wavfile_output.writeframes(output.tobytes())
