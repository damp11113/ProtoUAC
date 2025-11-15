import math
import traceback
import wave
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import threading
import time

from parametric_coding import PSEncoder, PSDecoder
from sbr import SBREncoder, SBRDecoder
from MDCT import mdct4, imdct4
from packer import SBRDataPacker, CompressionMode,pack_stereo_metadata, unpack_stereo_metadata, SBRDataUnpacker
from utils import apply_lowpass, detect_max_freq_response, design_lowpass_filter, vorbis_window

def spectral_complexity(audio, fs=48000, fmin=8000, fmax=16000):
    # FFT
    fft_data = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/fs)

    # Select 8–16 kHz band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = np.abs(fft_data[band_mask]) ** 2

    # Avoid log(0)
    band_power = np.maximum(band_power, 1e-12)

    # Compute spectral flatness
    geo_mean = np.exp(np.mean(np.log(band_power)))
    arith_mean = np.mean(band_power)
    sfm = geo_mean / arith_mean

    # Compute total power in that band (for energy check)
    band_energy_db = 10 * np.log10(np.mean(band_power))

    return sfm, band_energy_db


def process_chunk(chunk_data, chunk_id, progress_queue):
    """Process a single chunk of audio data in a separate process"""
    BBMaxFreq = 4000
    MaxFreq = 20000
    SBRPoins = 32
    PSminFreq = 150
    PSmaxFreq = 12000
    PSpoints = 160

    frame_size = 1024 * 2

    # SBR for Harmonic
    SBRencoder = SBREncoder(48000, BBMaxFreq, MaxFreq, SBRPoins, -80, 10, -80)
    SBRdecoder = SBRDecoder(48000, BBMaxFreq, MaxFreq, SBRPoins, chunk_size=frame_size)

    # Parametrix Stereo
    PSenc = PSEncoder(48000, PSminFreq, PSmaxFreq, PSpoints, -50, use_grouping=False)
    PSdec = PSDecoder(48000, PSminFreq, PSmaxFreq, PSpoints, use_grouping=False)

    hop_size = frame_size // 2

    # Create normalized window (Hann window with 50% overlap has perfect reconstruction)
    MDCT_window = vorbis_window(frame_size)

    # Create normalized window (Hann window with 50% overlap has perfect reconstruction)
    hann_window = np.hanning(frame_size)[:, np.newaxis]

    # For perfect reconstruction with 50% overlap, scale the window
    # sqrt(hann) gives constant overlap-add (COLA) property
    analysis_window = np.sqrt(hann_window)
    synthesis_window = analysis_window  # Same window for synthesis

    processed_frames = []
    prev_input_hop = None
    prev_overlap = np.zeros((hop_size, 2), dtype=np.float32)

    frame_count = 0
    total_hops = len(chunk_data) // hop_size

    dynamic_coding = False

    SBRpacker = SBRDataPacker(min_db=-80.0, max_db=0.0, delta_threshold=1.0)

    SBRunpacker = SBRDataUnpacker(min_db=-80.0, max_db=0.0)

    prev_frame = None
    prev_frame_hr = None

    for i in range(0, len(chunk_data), hop_size):
        hop_array = chunk_data[i:i + hop_size]

        if len(hop_array) == 0:
            break

        frame_count += 1

        # Build full frame from overlap
        if frame_count == 1:
            input_array = np.vstack((np.zeros((hop_size, 2), dtype=np.float32), hop_array))
        else:
            input_array = np.vstack((prev_input_hop, hop_array))

        prev_input_hop = hop_array.copy()

        N = len(input_array)

        # Zero pad if needed
        if N < frame_size:
            pad = np.zeros((frame_size - N, 2), dtype=np.float32)
            input_array = np.vstack((input_array, pad))
            N = frame_size

        # Apply analysis window ONCE

        windowed_input = input_array * analysis_window
        mono_audio_unwindowed = np.mean(input_array, axis=1)
        mono_audio = np.mean(windowed_input, axis=1)

        if dynamic_coding:
            side = (input_array[:, 0] - input_array[:, 1]) / 2
            side_fft_result = np.fft.rfft(side)
            side_freq = np.fft.rfftfreq(len(side), 1 / 48000)
            max_freq_stereo = detect_max_freq_response(side_fft_result, side_freq, -50)

            mid_fft_result = np.fft.rfft(mono_audio_unwindowed)
            mid_freq = np.fft.rfftfreq(len(mono_audio_unwindowed), 1 / 48000)
            max_freq_mid = detect_max_freq_response(mid_fft_result, mid_freq, -80)

            # enable SBR if needed
            if max_freq_mid > BBMaxFreq:
                useSBR = True
                SBRencoder.set_freq(BBMaxFreq, min(max_freq_mid, MaxFreq), SBRPoins)
            else:
                useSBR = False

            PSenc.set_freq(PSminFreq, min(max_freq_stereo, PSmaxFreq), PSpoints)
        else:
            useSBR = True

        # Stereo analysis
        stereo_profile = PSenc.analyze(input_array, True)
        pan_values = [pan for freq, pan, ipd, ic in stereo_profile]
        ipd_values = [ipd for freq, pan, ipd, ic in stereo_profile]
        ic_values = [ic >= 1 for freq, pan, ipd, ic in stereo_profile]

        # SBR analysis
        if useSBR:
            SBR_data, shouldUseNoises, is_transient = SBRencoder.analyze(mono_audio, True, True)
            packedSBR = SBRpacker.pack(SBR_data, is_transient, shouldUseNoises, CompressionMode.INT8)

        packedPS = pack_stereo_metadata(pan_values, ipd_values, ic_values, 0, 0, 0, len(stereo_profile))

        #print((len(packedSBR) + len(packedPS)) * 8)

        windowed_mono_input = MDCT_window * mono_audio
        mdct_coeffs = mdct4(windowed_mono_input)
        limited_coeffs = np.copy(mdct_coeffs)
        limited_coeffs[BBMaxFreq:] = 0

        # transmit parameters process implement on here

        # decoder side
        if dynamic_coding:
            PSdec.set_freq(PSminFreq, min(max_freq_stereo, PSmaxFreq), PSpoints)

            if useSBR:
                SBRdecoder.set_freq(BBMaxFreq, min(max_freq_mid, MaxFreq), SBRPoins)

        # Apply lowpass filter
        #lowpassed_audio = apply_lowpass(mono_audio, filter_b, filter_a) # simulate lossy
        lowpassed_audio_unwindowed = imdct4(limited_coeffs)  # No window here!

        # SBR decoding (on unwindowed signal)
        if useSBR:
            unpacked_energies_hr, unpacked_transient_hr, mode, shouldUseNoises = SBRunpacker.unpack(
                packedSBR,
                prev_frame_hr[0] if prev_frame_hr is not None else None,
                prev_frame_hr[1] if prev_frame_hr is not None else None
            )
            prev_frame_hr = (unpacked_energies_hr, shouldUseNoises)

            sbr_signal = SBRdecoder.generate(
                baseband_mono=lowpassed_audio_unwindowed,  # Feed unwindowed
                band_energies=unpacked_energies_hr,
                is_transient=unpacked_transient_hr,
                should_use_noises=shouldUseNoises,
                infloat=True
            )
        else:
            sbr_signal = np.zeros_like(lowpassed_audio_unwindowed)

        # Mix baseband and SBR at full frame_size
        output_mono_unwindowed = lowpassed_audio_unwindowed + sbr_signal * 0.5

        # Parametric Stereo on FULL frame
        p, ip, ic, minf, maxf, points = unpack_stereo_metadata(packedPS)

        reconstructed_stereo_unwindowed = PSdec.apply(
            mono_audio=output_mono_unwindowed,  # Full frame
            pan_values=p,
            ipd_values=ip,
            ic_values=ic
        )

        # Apply synthesis window to FULL frame
        reconstructed_stereo_windowed = reconstructed_stereo_unwindowed * synthesis_window

        # Overlap-add with FULL windowed frame
        output_frame = reconstructed_stereo_windowed[:hop_size] + prev_overlap
        prev_overlap = reconstructed_stereo_windowed[hop_size:].copy()

        processed_frames.append(output_frame)

        # Report progress via queue (every 10 frames to reduce overhead)
        if frame_count % 10 == 0:
            progress_queue.put(('progress', chunk_id, frame_count, total_hops))

    # Final update
    progress_queue.put(('progress', chunk_id, frame_count, total_hops))

    # Return processed frames and the last overlap for boundary stitching
    return chunk_id, processed_frames, prev_overlap


class ProgressTracker:
    """Thread-safe progress tracker for multiple chunks"""

    def __init__(self, num_chunks):
        self.num_chunks = num_chunks
        self.progress = {i: {'current': 0, 'total': 0, 'status': 'Waiting'} for i in range(num_chunks)}
        self.lock = threading.Lock()

    def update(self, chunk_id, current, total):
        with self.lock:
            self.progress[chunk_id]['current'] = current
            self.progress[chunk_id]['total'] = total
            self.progress[chunk_id]['status'] = 'Processing'

    def mark_complete(self, chunk_id):
        with self.lock:
            self.progress[chunk_id]['status'] = 'Complete'

    def display(self):
        with self.lock:
            lines = []
            for chunk_id in sorted(self.progress.keys()):
                p = self.progress[chunk_id]
                if p['total'] > 0:
                    percent = (p['current'] / p['total']) * 100
                    bar_length = 30
                    filled = int(bar_length * p['current'] / p['total'])
                    bar = '█' * filled + '░' * (bar_length - filled)
                    lines.append(f"Process {chunk_id + 1}: [{bar}] {percent:5.1f}% - {p['status']}")
                else:
                    lines.append(f"Process {chunk_id + 1}: [{'░' * 30}]   0.0% - {p['status']}")
            return '\n'.join(lines)


def progress_monitor(progress_queue, tracker, stop_event):
    """Monitor progress updates from worker processes"""
    while not stop_event.is_set():
        try:
            msg = progress_queue.get(timeout=0.1)
            if msg[0] == 'progress':
                _, chunk_id, current, total = msg
                tracker.update(chunk_id, current, total)
        except:
            continue


def main():
    win = wave.open(r"sample.wav", "rb")
    wout = wave.open(r"output.wav", "wb")
    wout.setnchannels(2)
    wout.setsampwidth(2)
    wout.setframerate(48000)

    # Read all audio data
    print("Reading audio data...")
    all_frames = win.readframes(win.getnframes())
    audio_data = np.frombuffer(all_frames, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
    win.close()

    # Split into chunks (with overlap for boundary handling)
    num_processes = 4  # Adjust based on your CPU cores
    chunk_size = len(audio_data) // num_processes
    overlap_size = 1024  # One frame overlap for continuity

    chunks = []
    for i in range(num_processes):
        start = max(0, i * chunk_size - overlap_size if i > 0 else 0)
        end = min(len(audio_data), (i + 1) * chunk_size + overlap_size)
        chunks.append((audio_data[start:end], i))

    print(f"Processing {num_processes} chunks with multiprocessing...")
    print(f"Using {num_processes} CPU cores\n")

    # Create progress tracker and queue
    tracker = ProgressTracker(num_processes)
    manager = Manager()
    progress_queue = manager.Queue()

    # Start progress monitoring thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_queue, tracker, stop_monitor),
        daemon=True
    )
    monitor_thread.start()

    # Start display thread
    stop_display = threading.Event()

    def display_progress():
        while not stop_display.is_set():
            print(f"\r{tracker.display()}", end='', flush=True)
            time.sleep(1)  # Update every 200ms

    display_thread = threading.Thread(target=display_progress, daemon=True)
    display_thread.start()

    # Process chunks in parallel using multiprocessing
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                process_chunk,
                chunk_data,
                chunk_id,
                progress_queue
            ): chunk_id
            for chunk_data, chunk_id in chunks
        }

        for future in as_completed(futures):
            chunk_id = futures[future]
            try:
                result = future.result()
                results[result[0]] = result
                tracker.mark_complete(chunk_id)
            except Exception as e:
                print(traceback.format_exc())
                print(f"\nProcess {chunk_id} failed: {e}")

    # Stop monitoring and display threads
    stop_monitor.set()
    stop_display.set()
    monitor_thread.join(timeout=1)
    display_thread.join(timeout=1)
    print(f"\n\n{tracker.display()}\n")

    # Merge results in order
    print("Merging chunks...")
    for chunk_id in sorted(results.keys()):
        _, processed_frames, last_overlap = results[chunk_id]

        for frame in processed_frames:
            output_data = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            wout.writeframes(output_data)

    wout.close()
    print("Processing complete!")


if __name__ == "__main__":
    main()