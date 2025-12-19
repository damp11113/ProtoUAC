import struct
import numpy as np
from scipy.io import wavfile
import sys
from phsc import HarmonicStereoExtractor, HarmonicStereoSynthesizer

input_filename = 'sample.wav'
output_filename = 'output.phsc.wav'

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

input_l = audio_float[:, 0]
input_r = audio_float[:, 1]

# Calculate the mono source signal
mono_source = (input_l + input_r) / 2.0

N = len(input_l)

# PHSC Configuration
W_SIZE = 2048  # Window size / Chunk size
HOP_SIZE = W_SIZE // 2  # 50% overlap

extractor = HarmonicStereoExtractor(
    sample_rate=sr, window_size=W_SIZE, hop_size=HOP_SIZE,
    min_f0_freq=120.0, max_f0_freq=2000.0, peak_threshold=0.0,
    max_harmonics_per_f0=5, max_harmonic_freq_object=5,
    log=True
)

synthesizer = HarmonicStereoSynthesizer(
    sample_rate=sr, window_size=W_SIZE, hop_size=HOP_SIZE, stereo_width=1.5
)

# Initialize Overlap-Add (OLA) buffer
output_l = np.zeros(N + W_SIZE, dtype=np.float32)
output_r = np.zeros(N + W_SIZE, dtype=np.float32)

print(f"\nStarting chunk processing (SR: {sr}, Chunk Size: {W_SIZE}, Hop Size: {HOP_SIZE})...")

def quantize_value(value, src_min, src_max, dst_min, dst_max):
    value = max(min(value, src_max), src_min)
    norm = (value - src_min) / (src_max - src_min)
    return int(round(dst_min + norm * (dst_max - dst_min)))


def dequantize_value(index, src_min, src_max, dst_min, dst_max):
    """
    Reverse of quantize_value().
    Convert integer index back into approximate float.
    """
    # Normalize 0..1
    norm = (index - dst_min) / (dst_max - dst_min)

    # Map back to original float space
    return src_min + norm * (src_max - src_min)

# old
#Qranges = {
#    "IID":  (-12.0, 12.0, -7, 7),      # 15-level AAC-style
#    "ICLD": (-12.0, 12.0, 0, 31),      # 32-level AAC-style
#    "IPD":  (-3.0,  3.0, 0, 7),        # radians → 0-7
#    "ICC":  (0.1,   1.0, 0, 7)         # correlation → 0-7
#}

Qranges = {
    "IID":  (-20.0, 20.0, -7, 7),      # 15-level AAC-style
    "ICLD": (-20.0, 20.0, 0, 31),      # 32-level AAC-style
    "IPD":  (-10.0,  10.0, 0, 7),        # radians → 0-7
    "ICC":  (0.1,   1.0, 0, 7)         # correlation → 0-7
}

QDranges = {
    "IID":  (-20.0, 20.0, -7, 7),      # 15-level AAC-style
    "ICLD": (-20.0, 20.0, 0, 31),      # 32-level AAC-style
    "IPD":  (-10.0,  10.0, 0, 7),        # radians → 0-7
    "ICC":  (0.1,   1.0, 0, 7)         # correlation → 0-7
}

def quantize_harmonic_objects(harmonic_objects):
    for obj in harmonic_objects:
        for h in obj["harmonics"]:
            for key in ["IID", "ICLD", "IPD", "ICC"]:
                src_min, src_max, dst_min, dst_max = Qranges[key]
                h[key] = quantize_value(h[key], src_min, src_max, dst_min, dst_max)

    return harmonic_objects


def dequantize_harmonic_objects(harmonic_objects):
    for obj in harmonic_objects:
        for h in obj["harmonics"]:
            for key in ["IID", "ICLD", "IPD", "ICC"]:
                src_min, src_max, dst_min, dst_max = QDranges[key]
                h[key] = dequantize_value(h[key], src_min, src_max, dst_min, dst_max)

    return harmonic_objects

def pack_harmonic_objects_full(harmonic_objects):
    out = bytearray()

    for obj in harmonic_objects:
        freq = int(obj["freq"])
        n = len(obj["harmonics"])

        # Header: freq (uint16) + n_harmonic (uint8)
        out += struct.pack(">H", freq)
        out += struct.pack(">B", n)

        # Harmonics: 2 bytes each
        for h in obj["harmonics"]:
            iid  = h["IID"]  & 0xF
            icld = h["ICLD"] & 0x1F
            ipd  = h["IPD"]  & 0x7
            icc  = h["ICC"]  & 0x7

            packed16 = (
                (iid   << 11) |
                (icld  << 6 ) |
                (ipd   << 3 ) |
                (icc)
            )

            out += struct.pack(">H", packed16)

    return bytes(out)

def unpack_harmonic_objects_full(data):
    idx = 0
    objects = []

    length = len(data)

    while idx < length:
        # -------- HEADER --------
        # freq: uint16
        freq = struct.unpack_from(">H", data, idx)[0]
        idx += 2

        # n_harmonic: uint8
        n_harm = struct.unpack_from(">B", data, idx)[0]
        idx += 1

        harmonics = []

        # -------- HARMONICS --------
        for _ in range(n_harm):
            # 16-bit packed harmonic
            packed16 = struct.unpack_from(">H", data, idx)[0]
            idx += 2

            trans = bool((packed16 >> 15) & 1)
            iid   = (packed16 >> 11) & 0xF
            icld  = (packed16 >> 6 ) & 0x1F
            ipd   = (packed16 >> 3 ) & 0x7
            icc   = packed16 & 0x7

            harmonics.append({
                "IID": iid,
                "ICLD": icld,
                "IPD": ipd,
                "ICC": icc
            })

        # -------- STORE OBJECT --------
        objects.append({
            "freq": freq,
            "n_harmonic": n_harm,
            "harmonics": harmonics
        })

    return objects

avgbitrate = []

# Processing loop
for i in range(0, N, HOP_SIZE):
    # 1. Get current stereo chunk and mono chunk
    chunk_end = i + W_SIZE

    # Extract input chunks
    chunk_l = input_l[i:chunk_end]
    chunk_r = input_r[i:chunk_end]
    chunk_mono = mono_source[i:chunk_end]

    # 2. Analysis: Extract PS parameters from the stereo chunk
    harmonic_objects = extractor.process_chunk(chunk_l, chunk_r)

    packed = pack_harmonic_objects_full(quantize_harmonic_objects(harmonic_objects))

    avgbitrate.append((len(packed) * 8) * (sr /W_SIZE))

    dequantized = dequantize_harmonic_objects(unpack_harmonic_objects_full(packed))

    # 3. Synthesis: Apply parameters to the mono chunk to get stereo output
    syn_l, syn_r = synthesizer.process_chunk(chunk_mono, dequantized)

    # 4. Overlap-Add (OLA) to the output buffer
    output_l[i:i + W_SIZE] += syn_l
    output_r[i:i + W_SIZE] += syn_r

    # Optional progress log
    if (i // HOP_SIZE) % 10 == 0:
        sys.stdout.write(f"\rProcessed {i // HOP_SIZE} chunks... ({(i / N * 100):.1f}%)")
        sys.stdout.flush()

print(f"avg bitrate is", np.mean(avgbitrate)/1000, "kbps")

# Trim the output signal to the original length (or slightly longer due to final window)
final_output_l = output_l[:N]
final_output_r = output_r[:N]

# Combine channels and convert back to int16
final_stereo_output = np.column_stack([final_output_l, final_output_r])
final_stereo_output = np.clip(final_stereo_output, -1.0, 1.0)
int16_output = (final_stereo_output * 32767).astype(np.int16)

# Write to file
wavfile.write(output_filename, sr, int16_output)
print(f"\rProcessing complete. Output written to: {output_filename}")
