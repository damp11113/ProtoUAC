import wave
import numpy as np
import scipy.signal as scsignal
from tqdm import tqdm
import SubPS

input_wav = './STD_TEST/sample_compressed.wav'
output_wav = "./Output/" + input_wav.split("/")[-1] + '_PS_output.wav'
frame_size = 256
resolution = "int8" # int8 int16
SubPS_bands = 32
SubPS_min_freq = 520
SubPS_max_freq = 8000
SubPS_SubFrame_scale = 8

wavfile_input = wave.open(input_wav, 'rb')
wavfile_output = wave.open(output_wav, 'wb')

sample_rate = wavfile_input.getframerate()
sample_width = wavfile_input.getsampwidth()

wavfile_output.setnchannels(2)
wavfile_output.setsampwidth(sample_width)
wavfile_output.setframerate(sample_rate)
total_frames = wavfile_input.getnframes()

CHUNK_SIZE = int((frame_size / 1000) * sample_rate)
hop_size = CHUNK_SIZE // 3

print(f"CHUNK_SIZE: {CHUNK_SIZE}")
print(f"hop_size: {hop_size}")
print(f"Encoder sub frame: {CHUNK_SIZE // SubPS_SubFrame_scale}")

subps_enc = SubPS.SubPSEncoder(sample_rate=sample_rate, num_bands=SubPS_bands, f_min=SubPS_min_freq, f_max=SubPS_max_freq, frame_size=CHUNK_SIZE // SubPS_SubFrame_scale, resolution=resolution)
subps_dec = SubPS.SubPSDecoder(sample_rate=sample_rate, num_bands=SubPS_bands, f_min=SubPS_min_freq, f_max=SubPS_max_freq, frame_size=CHUNK_SIZE // SubPS_SubFrame_scale, resolution=resolution)

signal = np.frombuffer(wavfile_input.readframes(total_frames), dtype=np.int16).reshape(-1, 2)

# Window function (Hamming)
window = scsignal.get_window('hamming', CHUNK_SIZE)

# Initialize the output signal as float64 to avoid type issues during processing
output_signal = np.zeros_like(signal, dtype=np.int16)

# Overlap-Add Processing (Without FFT)
start = 0
num_chunks = (len(signal) - CHUNK_SIZE) // hop_size + 1

# Add tqdm progress bar
for _ in tqdm(range(num_chunks), desc="Processing Chunks", unit="chunk"):
    # Extract the current chunk
    chunk = signal[start:start + CHUNK_SIZE]

    # Apply the window function
    left_channel = chunk[:, 0]
    right_channel = chunk[:, 1]

    # Apply the window to each channel
    left_windowed = left_channel * window
    right_windowed = right_channel * window

    # Stack them back together
    windowed_chunk = np.column_stack((left_windowed, right_windowed))

    # process is here
    mono_output, para_output = subps_enc.encode(windowed_chunk.astype(np.int16))

    print(f"para_output: {len(para_output)}")

    processed_data = subps_dec.decode(mono_output, para_output)

    # Overlap-add the windowed chunk to the output signal
    output_signal[start:start + CHUNK_SIZE] += processed_data

    # Move to the next chunk
    start += hop_size

# Convert output_signal back to int16 for audio playback
output_signal = np.clip(output_signal, -32768, 32767)

wavfile_output.writeframes(output_signal.astype(np.int16).tobytes())
wavfile_output.close()