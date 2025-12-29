import struct
import soundfile as sf
import logging
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from tqdm import tqdm
from mps_packer import packObj as MPSpackObj, unpackObj as MPSunpackObj
from hps_smoother import ParameterSmoother
from packer import pack_stereo_metadata, unpack_stereo_metadata
from parametric_coding import PSEncoder, PSDecoder

# create log with date info
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def process_chunk(chunk_data, sr, frame_size, hop_size, encoder_params, decoder_params, progress_dict):
    chunk_idx, stereo_chunk, mono_chunk, start_idx = chunk_data
    
    # Initialize progress for this chunk
    progress_dict[chunk_idx] = 0
    
    # Initialize encoder and decoder in this process
    PSenc = PSEncoder(sr, 150, 12000, 255, -50, use_grouping=False)
    PSdec = PSDecoder(sr, 150, 12000, 255, use_grouping=True, stereo_width=2.0)

    highDetailsMode = False
    pnbytes = 1

    smoother_mps = ParameterSmoother(alpha=0.5, method='lerp')
    
    # Process this chunk
    n_samples = len(stereo_chunk)
    output = np.zeros_like(stereo_chunk)
    hann_window = np.hanning(frame_size)[:, np.newaxis]
    prev_overlap = np.zeros((hop_size, 2), dtype=np.float32)
    
    # Calculate total frames for progress tracking
    total_frames = (n_samples + hop_size - 1) // hop_size
    processed_frames = 0
    avgbitrate = []    
    
    for i in range(0, n_samples, hop_size):
        frame_end = min(i + frame_size, n_samples)
        
        # Get current stereo and mono frames
        stereo_frame = stereo_chunk[i:frame_end]
        mono_frame = mono_chunk[i:frame_end]
        
        # Pad if needed
        if len(stereo_frame) < frame_size:
            stereo_frame = np.pad(
                stereo_frame,
                ((0, frame_size - len(stereo_frame)), (0, 0)),
                mode='constant'
            )
            mono_frame = np.pad(
                mono_frame,
                (0, frame_size - len(mono_frame)),
                mode='constant'
            )
        
        # encoder side
        stereo_profile = PSenc.analyze(stereo_frame, True)
        
        if highDetailsMode:
            pan_values = [pan for freq, pan, ipd, ic in stereo_profile]
            ipd_values = [ipd for freq, pan, ipd, ic in stereo_profile]
            ic_values = [ic >= 1 for freq, pan, ipd, ic in stereo_profile]
            packedPS = pack_stereo_metadata(pan_values, ipd_values, ic_values, 0, 0, 0, len(stereo_profile))
        else:
            packedPS = MPSpackObj(stereo_profile, 0, 0, 0, nbytes=pnbytes)


        avgbitrate.append((len(packedPS) * 8) * (sr / frame_size))

        # decoder side
        if highDetailsMode:
            p, ip, ic, _, _, _ = unpack_stereo_metadata(packedPS)
        else:
            decodedI, _, _, _ = MPSunpackObj(packedPS)
            smoothedI = smoother_mps.smooth_mps_results(decodedI)

            p = [pan for freq, pan, ipd, ic in smoothedI]
            ip = [ipd for freq, pan, ipd, ic in smoothedI]
            ic = [ic >= 1 for freq, pan, ipd, ic in smoothedI]

        # Synthesis: Apply parameters to mono
        reconstructed = PSdec.apply(
            mono_audio=mono_frame,
            pan_values=p,
            ipd_values=ip,
            ic_values=ic
        )

        reconstructed *= hann_window
        
        # Overlap-add
        output_frame = reconstructed[:hop_size] + prev_overlap
        prev_overlap = reconstructed[hop_size:].copy()
        
        # Calculate how much space is left in buffer
        remaining_samples = n_samples - i
        samples_to_write = min(hop_size, remaining_samples)
        
        # Store processed frame
        output[i:i+samples_to_write] += output_frame[:samples_to_write]
        
        # Update progress
        processed_frames += 1
        progress_dict[chunk_idx] = int((processed_frames / total_frames) * 100)
    
    logging.info(f"Chunk {chunk_idx} completed (samples {start_idx} to {start_idx + n_samples})")
    print(f"avg bitrate is", np.mean(avgbitrate)/1000, "kbps")
    progress_dict[chunk_idx] = 100
    
    return (chunk_idx, output, start_idx)

def main():
    # Load audio
    logging.info("Loading audio file...")
    stereo_audio, sr = sf.read('sample.wav')
    mono_audio = np.mean(stereo_audio, axis=1)
    
    # Parameters
    frame_size = 1024 * 3
    hop_size = frame_size // 2
    
    # Encoder parameters
    encoder_params = {
        'min_freq': 120.0,
        'max_freq': 14000,
        'imp_point': 8,
        'log_scale': True,
        'hps_L_h': 0.1,
        'hps_L_p': 1000,
        'max_harmonic_freq_object': 7,
        'max_harmonics_per_f0': 3
    }
    
    # Decoder parameters
    decoder_params = {
        'min_freq': 120.0,
        'max_freq': 14000,
        'imp_point': 8,
        'log_scale': True,
        'hps_L_h': 0.1,
        'hps_L_p': 1000,
        'stereo_width': 1.5
    }
    
    # Determine number of processes
    n_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    logging.info(f"Using {n_processes} processes")
    
    # Split audio into chunks for multiprocessing
    # Each chunk should be large enough to amortize process creation overhead
    # but small enough to balance load across processes
    n_samples = len(stereo_audio)
    chunk_duration = 10.0  # seconds per chunk
    chunk_size = int(chunk_duration * sr)
    
    # Add overlap to avoid artifacts at chunk boundaries
    overlap_size = frame_size * 2
    
    chunks = []
    chunk_idx = 0
    
    for i in range(0, n_samples, chunk_size):
        start = max(0, i - overlap_size)
        end = min(n_samples, i + chunk_size + overlap_size)
        
        stereo_chunk = stereo_audio[start:end]
        mono_chunk = mono_audio[start:end]
        
        chunks.append((chunk_idx, stereo_chunk, mono_chunk, start))
        chunk_idx += 1
    
    logging.info(f"Split audio into {len(chunks)} chunks")
    
    # Create shared progress dictionary
    manager = Manager()
    progress_dict = manager.dict()
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_chunk,
        sr=sr,
        frame_size=frame_size,
        hop_size=hop_size,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        progress_dict=progress_dict
    )
    
    # Process chunks in parallel with progress bar
    logging.info("Processing chunks in parallel...")
    with Pool(processes=n_processes) as pool:
        # Start async processing
        async_result = pool.map_async(process_func, chunks)
        
        # Monitor progress
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            completed = 0
            while not async_result.ready():
                # Count completed chunks (100% progress)
                current_completed = sum(1 for v in progress_dict.values() if v == 100)
                if current_completed > completed:
                    pbar.update(current_completed - completed)
                    completed = current_completed
                
                # Show individual chunk progress
                progress_str = " | ".join([f"C{k}:{v}%" for k, v in sorted(progress_dict.items()) if v < 100])
                if progress_str:
                    pbar.set_postfix_str(progress_str[:50])  # Limit length
                
                async_result.wait(0.1)
            
            # Final update
            pbar.update(len(chunks) - completed)
        
        results = async_result.get()
    
    # Sort results by chunk index
    results.sort(key=lambda x: x[0])
    
    # Merge results with crossfade for smooth transitions
    logging.info("Merging processed chunks with crossfade...")
    output = np.zeros_like(stereo_audio)
    
    # Crossfade length (in samples)
    crossfade_length = overlap_size
    
    for idx, (chunk_idx, processed_chunk, start_idx) in enumerate(results):
        end_idx = min(start_idx + len(processed_chunk), n_samples)
        actual_length = end_idx - start_idx
        
        if idx == 0:
            # First chunk: no crossfade at the beginning
            output[start_idx:end_idx] = processed_chunk[:actual_length]
        else:
            # Get previous chunk info
            prev_chunk_idx, prev_processed, prev_start = results[idx - 1]
            prev_end = min(prev_start + len(prev_processed), n_samples)
            
            # Check if there's overlap
            overlap_start = max(start_idx, prev_start)
            overlap_end = min(end_idx, prev_end)
            
            if overlap_start < overlap_end:
                # There is overlap - apply crossfade
                overlap_len = overlap_end - overlap_start
                fade_len = min(overlap_len, crossfade_length)
                
                # Create fade curve (cosine crossfade for smoothness)
                fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
                fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
                
                # Apply fade to overlap region
                # Calculate positions in each chunk
                prev_overlap_start = overlap_start - prev_start
                curr_overlap_start = overlap_start - start_idx
                
                # Only fade the first part of overlap
                fade_region_end = overlap_start + fade_len
                
                # Fade out the end of previous chunk
                prev_fade_start = prev_overlap_start
                prev_fade_end = prev_fade_start + fade_len
                output[overlap_start:fade_region_end] = (
                    output[overlap_start:fade_region_end] * fade_out[:, np.newaxis]
                )
                
                # Fade in the beginning of current chunk and add
                curr_fade_start = curr_overlap_start
                curr_fade_end = curr_fade_start + fade_len
                output[overlap_start:fade_region_end] += (
                    processed_chunk[curr_fade_start:curr_fade_end] * fade_in[:, np.newaxis]
                )
                
                # After crossfade region, just use current chunk
                if fade_region_end < end_idx:
                    non_overlap_start = max(fade_region_end, prev_end)
                    if non_overlap_start < end_idx:
                        offset = non_overlap_start - start_idx
                        length = end_idx - non_overlap_start
                        output[non_overlap_start:end_idx] = processed_chunk[offset:offset + length]
            else:
                # No overlap - just copy
                output[start_idx:end_idx] = processed_chunk[:actual_length]
    
    logging.info("Crossfade merging complete")
    
    # Save output
    logging.info("Saving output file...")
    sf.write('output.mps.wav', output, sr)
    logging.info("Multiprocess processing complete!")


if __name__ == '__main__':
    main()