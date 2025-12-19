import numpy as np
from scipy.io import wavfile
import sys
from parametric_coding import PSEncoder, PSDecoder

input_filename = 'sample.wav'
output_filename = 'output.psc.wav'

try:
    sr, audio_data = wavfile.read(input_filename)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_filename}")
    exit()
except Exception as e:
    print(f"Error reading WAV file: {e}")
    exit()

# Normalize audio data to float32 [-1, 1]
if audio_data.dtype == np.int16:
    audio_float = audio_data.astype(np.float32) / 32768.0
else:
    audio_float = audio_data.astype(np.float32)

# Ensure stereo data
if audio_float.ndim < 2:
    print("Error: Input WAV file must be stereo (2 channels).")
    exit()

N = len(audio_float)

# PHSC Configuration
W_SIZE = 4096  # Window size / Chunk size
HOP_SIZE = W_SIZE // 2  # 50% overlap

PSenc = PSEncoder(sr, 20, 15000, 32, -50, use_grouping=False)
PSdec = PSDecoder(sr, 20, 15000, 32, use_grouping=True)

# Initialize Overlap-Add (OLA) buffer
output = np.zeros(N + W_SIZE, dtype=np.float32).reshape(-1, 2)

print(f"\nStarting chunk processing (SR: {sr}, Chunk Size: {W_SIZE}, Hop Size: {HOP_SIZE})...")

# Processing loop
# Processing loop
for i in range(0, N, HOP_SIZE):
    chunk_end = i + W_SIZE
    chunk = audio_float[i:chunk_end]

    # Zero-pad last chunk
    if chunk.shape[0] < W_SIZE:
        pad_len = W_SIZE - chunk.shape[0]
        chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant')

    mono_audio = np.mean(chunk, axis=1)

    stereo_profile = PSenc.analyze(chunk, True)
    pan_values = [pan for freq, pan, ipd, ic in stereo_profile]
    ipd_values = [ipd for freq, pan, ipd, ic in stereo_profile]
    ic_values = [ic >= 1 for freq, pan, ipd, ic in stereo_profile]

    reconstructed_stereo_unwindowed = PSdec.apply(
        mono_audio=mono_audio,
        pan_values=pan_values,
        ipd_values=ipd_values,
        ic_values=ic_values
    )

    # OLA with bounds checking
    output_end = min(i + W_SIZE, len(output))
    actual_len = output_end - i
    
    # Skip if we're beyond the buffer
    if actual_len <= 0:
        break
        
    output[i:output_end] += reconstructed_stereo_unwindowed[:actual_len]

    # Optional progress log
    if (i // HOP_SIZE) % 10 == 0:
        sys.stdout.write(f"\rProcessed {i // HOP_SIZE} chunks... ({(i / N * 100):.1f}%)")
        sys.stdout.flush()
        
# Trim the output signal to the original length (or slightly longer due to final window)
final_stereo_output = output[:N]

# Combine channels and convert back to int16
final_stereo_output = np.clip(final_stereo_output, -1.0, 1.0)
int16_output = (final_stereo_output * 32767).astype(np.int16)

# Write to file
wavfile.write(output_filename, sr, int16_output)
print(f"\rProcessing complete. Output written to: {output_filename}")
