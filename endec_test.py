import math
import traceback
import wave
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import threading
import time

from parametric_coding import PSEncoder, PSDecoder
from sbr import SBRDecoder, SBREncoder, SBRDecoderHR
from packer import SBRDataPacker, CompressionMode,pack_stereo_metadata, unpack_stereo_metadata, SBRDataUnpacker
from utils import apply_lowpass, detect_max_freq_response, design_lowpass_filter


def process_chunk(chunk_data, chunk_id, filter_params, progress_queue):
    """Process a single chunk of audio data in a separate process"""
    # SBR for High Freq
    SBRencoder = SBREncoder(48000, 16000, 22000, 8, -50, 10, -50)
    SBRdecoder = SBRDecoder(48000, 16000, 22000, 8)

    # SBR for Harmonic
    SBRencoderHR = SBREncoder(48000, 8000, 16000, 16, -50, 10, -50)
    SBRdecoderHR = SBRDecoderHR(48000, 8000, 16000, 16)

    # Parametrix Stereo
    PSenc = PSEncoder(48000, 150, 14000, 200, -75)
    PSdec = PSDecoder(48000, 150, 14000, 200)

    filter_b, filter_a = filter_params

    frame_size = 2048
    hop_size = frame_size // 2

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

    SBRpacker = SBRDataPacker(min_db=-50.0, max_db=0.0, delta_threshold=1.0)
    SBRpackerHR = SBRDataPacker(min_db=-50.0, max_db=0.0, delta_threshold=1.0)

    SBRunpacker = SBRDataUnpacker(min_db=-50.0, max_db=0.0)
    SBRunpackerHR = SBRDataUnpacker(min_db=-50.0, max_db=0.0)

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
        mono_audio = np.mean(windowed_input, axis=1)

        if dynamic_coding:
            side = (input_array[:, 0] - input_array[:, 1]) / 2
            side_fft_result = np.fft.rfft(side)
            side_freq = np.fft.rfftfreq(len(side), 1 / 48000)
            max_freq_stereo = detect_max_freq_response(side_fft_result, side_freq, -30)

            mid_fft_result = np.fft.rfft(mono_audio)
            mid_freq = np.fft.rfftfreq(len(mono_audio), 1 / 48000)
            max_freq_mid = detect_max_freq_response(mid_fft_result, mid_freq, -50)

            PSenc.set_freq(150, max_freq_stereo, 256)
            SBRencoder.set_freq(8000, min(max_freq_mid, 13500), 32)

        # Stereo analysis
        stereo_profile = PSenc.analyze(input_array, True)
        pan_values = [pan for freq, pan, ipd, ic in stereo_profile]
        ipd_values = [ipd for freq, pan, ipd, ic in stereo_profile]
        ic_values = [ic >= 1 for freq, pan, ipd, ic in stereo_profile]

        # SBR analysis
        SBRHF_data, is_HF_transient = SBRencoder.analyze(mono_audio, True)
        SBRHR_data, is_HR_transient = SBRencoderHR.analyze(mono_audio, True)

        packedSBR = SBRpacker.pack(SBRHF_data, is_HF_transient, CompressionMode.INT8)
        packedSBRHR = SBRpackerHR.pack(SBRHR_data, is_HR_transient, CompressionMode.INT8)

        packedPS = pack_stereo_metadata(pan_values, ipd_values, ic_values, 0, 0, 0, len(stereo_profile))

        #print((len(packedSBR) + len(packedPS)) * 8)

        # decoder side
        if dynamic_coding:
            PSdec.set_freq(150, max_freq_stereo, 256)
            SBRdecoder.set_freq(8000, min(max_freq_mid, 13500), 32)

        # Apply lowpass filter
        lowpassed_audio = apply_lowpass(mono_audio, filter_b, filter_a) # simulate lossy

        unpacked_energies, unpacked_transient, mode = SBRunpacker.unpack(packedSBR, prev_frame)
        prev_frame = unpacked_energies

        unpacked_energies_hr, unpacked_transient_hr, mode = SBRunpackerHR.unpack(packedSBRHR, prev_frame_hr)
        prev_frame_hr = unpacked_energies_hr

        # SBR decoding
        sbr_signal = SBRdecoder.generate(
            frame_length=len(mono_audio),
            band_energies=unpacked_energies,
            is_transient=unpacked_transient
        )

        sbrhr_signal = SBRdecoderHR.generate(
            baseband_mono=lowpassed_audio,
            band_energies=unpacked_energies_hr,
            is_transient=unpacked_transient_hr,
            infloat=True
        )

        # Mix
        min_len = min(len(lowpassed_audio), len(sbr_signal))
        output_mono = lowpassed_audio[:min_len] + sbrhr_signal[:min_len] * 1.2 + sbr_signal[:min_len] * 0.2

        p, ip, ic, minf, maxf, points = unpack_stereo_metadata(packedPS)

        # Reconstruct stereo
        reconstructed_stereo = PSdec.apply(
            mono_audio=output_mono,
            pan_values=p,
            ipd_values=ip,
            ic_values=ic
        )

        # Apply synthesis window ONCE (matching analysis window)
        reconstructed_stereo = reconstructed_stereo * synthesis_window

        # Overlap-add with previous overlap
        output_frame = reconstructed_stereo[:hop_size] + prev_overlap
        prev_overlap = reconstructed_stereo[hop_size:].copy()

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

    # Configuration
    SBR_CUTOFF = 8000

    print(f"Designing lowpass filter at {SBR_CUTOFF} Hz...")
    filter_b, filter_a = design_lowpass_filter(SBR_CUTOFF, 48000, order=8)
    filter_params = (filter_b, filter_a)

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
                filter_params,
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