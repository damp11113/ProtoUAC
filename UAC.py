import wave
import numpy as np
from scipy.signal import resample, medfilt, windows
from tqdm import tqdm  # Import tqdm for progress bar
from MDCT import mdct, imdct, mdct4_mono, imdct4_mono
from Subband import SubbandEncoder, SubbandDecoder
from Quantizator import DQInt8Complex, QComplexInt8

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

class UAC_LC_Encoder:
    def __init__(self):
