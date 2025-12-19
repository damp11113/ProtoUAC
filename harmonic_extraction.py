import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import medfilt, windows, stft, istft

def make_integer_odd(n):
    if n % 2 == 0:
        n += 1
    return n

def hps(x, Fs, N, H, L_h, L_p, L_unit='physical', mask='binary', detail=False, eps=0.001):
    """Harmonic-percussive separation (HPS) algorithm

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate of x
        N (int): Frame length
        H (int): Hopsize
        L_h (float): Horizontal median filter length given in seconds or frames
        L_p (float): Percussive median filter length given in Hertz or bins
        L_unit (str): Adjusts unit, either 'pyhsical' or 'indices' (Default value = 'physical')
        mask (str): Either 'binary' or 'soft' (Default value = 'binary')
        eps (float): Parameter used in soft maskig (Default value = 0.001)
        detail (bool): Returns detailed information (Default value = False)

    Returns:
        x_h (np.ndarray): Harmonic signal
        x_p (np.ndarray): Percussive signal
        details (dict): Dictionary containing detailed information; returned if ``detail=True``
    """
    assert L_unit in ['physical', 'indices']
    assert mask in ['binary', 'soft']
    # stft
    _, _, X = stft(x, fs=Fs, nperseg=N, noverlap=H, window='hann')
    # power spectrogram
    Y = np.abs(X) ** 2
    # median filtering
    if L_unit == 'physical':
        L_h = int(np.ceil(L_h * Fs / H))
        L_p = int(np.ceil(L_p * N / Fs))

    L_h = make_integer_odd(L_h)
    L_p = make_integer_odd(L_p)
    Y_h = medfilt(Y, [1, L_h])
    Y_p = medfilt(Y, [L_p, 1])

    # masking
    if mask == 'binary':
        M_h = np.int8(Y_h >= Y_p)
        M_p = np.int8(Y_h < Y_p)
    if mask == 'soft':
        M_h = (Y_h + eps / 2) / (Y_h + Y_p + eps)
        M_p = (Y_p + eps / 2) / (Y_h + Y_p + eps)

    X_h = X * M_h
    X_p = X * M_p

    # istft
    _, x_h = istft(X_h, fs=Fs, nperseg=N, noverlap=H, window='hann')
    _, x_p = istft(X_p, fs=Fs, nperseg=N, noverlap=H, window='hann')

    if detail:
        return x_h, x_p, dict(Y_h=Y_h, Y_p=Y_p, M_h=M_h, M_p=M_p, X_h=X_h, X_p=X_p)
    else:
        return x_h, x_p

def hps_stereo(x, Fs, N, H, L_h, L_p, L_unit='physical', mask='binary', detail=False, eps=0.001):
    # Check if signal is stereo (2 channels)
    # Standard shape is (channels, samples) or (samples, channels)
    # We ensure it's (channels, samples) for processing
    is_stereo = x.ndim > 1
    if not is_stereo:
        x_list = [x]
    else:
        # If input is (samples, 2), transpose it to (2, samples)
        if x.shape[0] > x.shape[1]:
            x = x.T
        x_list = x

    harmonic_channels = []
    percussive_channels = []

    # Process each channel independently
    for channel in x_list:
        # --- Original HPS Logic Start ---
        _, _, X = stft(channel, fs=Fs, nperseg=N, noverlap=H, window='hann')
        Y = np.abs(X) ** 2
        
        if L_unit == 'physical':
            actual_L_h = int(np.ceil(L_h * Fs / H))
            actual_L_p = int(np.ceil(L_p * N / Fs))
        else:
            actual_L_h, actual_L_p = L_h, L_p

        actual_L_h = make_integer_odd(actual_L_h)
        actual_L_p = make_integer_odd(actual_L_p)
        
        Y_h = medfilt(Y, [1, actual_L_h])
        Y_p = medfilt(Y, [actual_L_p, 1])

        if mask == 'binary':
            M_h = np.int8(Y_h >= Y_p)
            M_p = np.int8(Y_h < Y_p)
        else:
            M_h = (Y_h + eps / 2) / (Y_h + Y_p + eps)
            M_p = (Y_p + eps / 2) / (Y_h + Y_p + eps)

        _, x_h = istft(X * M_h, fs=Fs, nperseg=N, noverlap=H, window='hann')
        _, x_p = istft(X * M_p, fs=Fs, nperseg=N, noverlap=H, window='hann')
        # --- Original HPS Logic End ---

        harmonic_channels.append(x_h)
        percussive_channels.append(x_p)

    # Reconstruct stereo signals
    if is_stereo:
        # Stack back to (samples, channels) for standard audio playback
        final_h = np.stack(harmonic_channels, axis=-1)
        final_p = np.stack(percussive_channels, axis=-1)
    else:
        final_h = harmonic_channels[0]
        final_p = percussive_channels[0]

    return final_h, final_p

def harmonic_percussive_separation(input_wav, output_harmonic, output_percussive):
    """
    Fast Harmonic-Percussive Separation on a mono WAV file.

    Parameters:
    - input_wav: Path to input mono WAV file
    - output_harmonic: Path for output harmonic component WAV
    - output_percussive: Path for output percussive component WAV
    - kernel_size: Median filter kernel size (default 17 for speed)
    - beta: Softness parameter (higher = softer masks)
    """

    # Read the input WAV file
    sample_rate, audio = wavfile.read(input_wav)

    # Ensure mono
    #if len(audio.shape) > 1:
    #    audio = audio[:, 0]

    # Normalize to float32
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    harmonic_audio, percussive_audio = hps_stereo(audio, Fs=sample_rate, N=1024, H=512, L_h=0.1, L_p=1000)

    # Normalize and convert to int16
    h_max = np.max(np.abs(harmonic_audio))
    if h_max > 0:
        harmonic_audio = np.int16(harmonic_audio / h_max * 32767)
    else:
        harmonic_audio = np.int16(harmonic_audio)

    p_max = np.max(np.abs(percussive_audio))
    if p_max > 0:
        percussive_audio = np.int16(percussive_audio / p_max * 32767)
    else:
        percussive_audio = np.int16(percussive_audio)

    # Write output files
    wavfile.write(output_harmonic, sample_rate, harmonic_audio)
    wavfile.write(output_percussive, sample_rate, percussive_audio)

    print(f"Separation complete!")
    print(f"Harmonic component saved to: {output_harmonic}")
    print(f"Percussive component saved to: {output_percussive}")


# Example usage
if __name__ == "__main__":
    import time

    # Replace with your file paths
    input_file = r"sample.wav"
    harmonic_output = r"output.h.hx.wav"
    percussive_output = r"output.i.hx.wav"

    start = time.time()
    harmonic_percussive_separation(
        input_file,
        harmonic_output,
        percussive_output
    )
    print(f"Processing time: {time.time() - start:.2f} seconds")