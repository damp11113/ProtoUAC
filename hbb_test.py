import soundfile as sf
import logging
import numpy as np
from hbb import HyBBEncoder, HyBBDecoder
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from tqdm import tqdm

# create log with date info
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def process_chunk(chunk_data, sr, frame_size, hop_size, progress_dict):
    chunk_idx, stereo_chunk, mono_chunk, start_idx = chunk_data
    
    progress_dict[chunk_idx] = 0
    
    encoder = HyBBEncoder(
        sample_rate=sr, frame_size=frame_size, hop_size=hop_size,
        min_freq=20.0, max_freq=8000.0, min_freq_imp=250.0,
        max_freq_imp=8000, imp_point=32, max_harmonics_per_f0=10,
        max_harmonic_freq_object=10, log_scale=True, hps_L_h=0.1, hps_L_p=1000
    )
    
    decoder = HyBBDecoder(
        sample_rate=sr, frame_size=frame_size, hop_size=hop_size,
        min_freq_imp=250.0, max_freq_imp=8000, imp_point=32
    )
    
    n_samples = len(mono_chunk)
    output = np.zeros(n_samples, dtype=np.float32)
    
    # Track frames for progress
    total_frames = (n_samples + hop_size - 1) // hop_size
    processed_frames = 0
    
    # Initialize overlap buffer based on actual decoder output size
    # We'll determine the size after the first decode to be safe
    prev_overlap = None 
    
    for i in range(0, n_samples, hop_size):
        frame_end = min(i + frame_size, n_samples)
        mono_frame = mono_chunk[i:frame_end]
        
        if len(mono_frame) < frame_size:
            mono_frame = np.pad(mono_frame, (0, frame_size - len(mono_frame)), mode='constant')
        
        # Analysis and Synthesis
        harmonic_sd, impulse_sd = encoder.encode(mono_frame)
        reconstructed = decoder.decode(harmonic_sd, impulse_sd)
        
        # Flatten reconstructed if it comes out as (N, 1)
        reconstructed = reconstructed.flatten()
        
        recon_len = len(reconstructed)
        current_hop = recon_len // 2 # Assuming 50% overlap-add
        
        # Initialize or resize overlap buffer if needed
        if prev_overlap is None:
            prev_overlap = np.zeros(current_hop, dtype=np.float32)

        # Apply Hanning window to the reconstructed frame
        recon_windowed = reconstructed * np.hanning(recon_len)
        
        # Overlap-Add
        # Current frame's first half + previous frame's second half
        output_frame = recon_windowed[:current_hop] + prev_overlap
        prev_overlap = recon_windowed[current_hop:].copy()
        
        # Write to output buffer
        remaining = n_samples - i
        samples_to_write = min(current_hop, remaining)
        output[i:i+samples_to_write] = output_frame[:samples_to_write]
        
        processed_frames += 1
        progress_dict[chunk_idx] = int((processed_frames / total_frames) * 100)
    
    logging.info(f"Chunk {chunk_idx} completed")
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
    output = np.zeros_like(mono_audio)
    
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
                
                # Fade out the end of previous chunk (mono) and add faded current chunk
                prev_fade_start = prev_overlap_start
                prev_fade_end = prev_fade_start + fade_len
                # Elementwise multiply 1D arrays
                output[overlap_start:fade_region_end] *= fade_out

                # Fade in the beginning of current chunk and add
                curr_fade_start = curr_overlap_start
                curr_fade_end = curr_fade_start + fade_len
                output[overlap_start:fade_region_end] += (
                    processed_chunk[curr_fade_start:curr_fade_end] * fade_in
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
    sf.write('output.hbb.wav', output, sr)
    logging.info("Multiprocess processing complete!")


if __name__ == '__main__':
    main()