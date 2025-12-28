from scipy.io.wavfile import read, write
from multiprocessing import Pool, cpu_count
from functools import partial
from packer import HarmonicPacker, HarmonicUnpacker
from phxc import HarmonicExtractor, HarmonicGenerator
import numpy as np

def process_single_chunk(args, extractor_params, packer_params):
    """
    Process a single chunk in a separate process.
    Returns the chunk index and encoded data.
    """

    chunk_idx, chunk_data, sample_rate = args
    
    # Create extractor for this process
    extractor = HarmonicExtractor(
        sample_rate=sample_rate,
        window_size=extractor_params['window_size'],
        hop_size=extractor_params['hop_size'],
        min_f0_freq=extractor_params['min_f0_freq'],
        max_f0_freq=extractor_params['max_f0_freq'],
        peak_threshold=extractor_params['peak_threshold'],
        max_harmonics_per_f0=extractor_params['max_harmonics_per_f0'],
        max_harmonic_freq_output=extractor_params['max_harmonic_freq_output'],
        max_harmonic_freq_object=extractor_params['max_harmonic_freq_object']
    )
    
    # Create packer for this process
    packer = HarmonicPacker(
        sample_rate=sample_rate,
        window_size=packer_params['window_size'],
        amp_dtype=packer_params['amp_dtype'],
        phase_dtype=packer_params['phase_dtype'],
        scale_mode=packer_params['scale_mode'],
        prediction_mode="delta"
    )
    
    # Process the chunk
    harmonic_chunk = extractor.process_chunk(chunk_data)
    encoded = packer.pack_chunk(harmonic_chunk)
    
    return chunk_idx, encoded, len(harmonic_chunk)

def synthesize_single_chunk(args, generator_params, packer_params):
    """
    Synthesize a single chunk in a separate process.
    Returns the chunk index and synthesized audio.
    """    
    chunk_idx, encoded_data, sample_rate = args
    
    # Create generator for this process
    generator = HarmonicGenerator(
        sample_rate=sample_rate,
        window_size=generator_params['window_size'],
        hop_size=generator_params['hop_size']
    )
    
    # Create packer for this process
    unpacker = HarmonicUnpacker(
        sample_rate=sample_rate,
        window_size=packer_params['window_size'],
        amp_dtype=packer_params['amp_dtype'],
        phase_dtype=packer_params['phase_dtype'],
        scale_mode=packer_params['scale_mode']
    )
    
    # Decode and synthesize
    decoded_objects = unpacker.unpack_chunk(encoded_data)
    synthesized_chunk = generator.process_chunk(decoded_objects)
    synthesized_chunk = synthesized_chunk * generator.window
    
    return chunk_idx, synthesized_chunk

def process_chunk_by_chunk_multiprocessing(input_audio_path: str, output_audio_path: str, 
                                           params: dict, num_processes: int = None):
    """
    Reads a WAV file chunk by chunk, extracts Chunks using multiprocessing,
    then generates the output audio using multiprocessing and Overlap-Add method.
    """
        
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"--- Starting Analysis of {input_audio_path} with {num_processes} processes ---")

    try:
        sample_rate, input_data = read(input_audio_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_audio_path}")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Data type conversion and normalization
    if input_data.ndim > 1:
        input_data = input_data.mean(axis=1)
    if input_data.dtype.kind == 'i':
        max_val = np.iinfo(input_data.dtype).max
        audio_float = input_data.astype(np.float32) / max_val
    else:
        audio_float = input_data.astype(np.float32)

    if audio_float.size == 0:
        print("Input audio data is empty.")
        return

    # --- Setup Parameters ---
    WINDOW_SIZE = params['window_size']
    HOP_SIZE = params['hop_size']

    # Prepare extractor parameters
    extractor_params = {
        'window_size': WINDOW_SIZE,
        'hop_size': HOP_SIZE,
        'min_f0_freq': params['min_f0_freq'],
        'max_f0_freq': params['max_f0_freq'],
        'peak_threshold': params['peak_threshold'],
        'max_harmonics_per_f0': params['max_harmonics_per_f0'],
        'max_harmonic_freq_output': params['max_harmonic_freq_output'],
        'max_harmonic_freq_object': params['max_harmonic_freq_object']
    }
    
    # Prepare packer parameters
    packer_params = {
        'window_size': WINDOW_SIZE,
        'amp_dtype': 'uint8',
        'phase_dtype': 'uint8',
        'scale_mode': 'log'
    }

    # --- Prepare chunks for processing ---
    print("Preparing chunks for analysis...")
    chunk_args = []
    chunk_idx = 0
    
    for i in range(0, audio_float.size, HOP_SIZE):
        chunk_start = i
        chunk_end = i + WINDOW_SIZE
        current_chunk = audio_float[chunk_start:chunk_end]
        
        if current_chunk.size == 0:
            break
        
        chunk_args.append((chunk_idx, current_chunk, sample_rate))
        chunk_idx += 1
    
    total_chunks = len(chunk_args)
    print(f"Total chunks to process: {total_chunks}")

    # --- Analysis with Multiprocessing ---
    print(f"Extracting chunks using {num_processes} processes...")
    
    process_func = partial(process_single_chunk, 
                          extractor_params=extractor_params,
                          packer_params=packer_params)
    
    extracted_parameters = [None] * total_chunks
    maxvalue = 0
    avgbitrate = []
    
    with Pool(processes=num_processes) as pool:
        results = pool.imap(process_func, chunk_args, chunksize=10)
        
        for chunk_idx, encoded, num_sources in results:
            extracted_parameters[chunk_idx] = encoded
            maxvalue = max(maxvalue, len(encoded) * 8)

            avgbitrate.append(len(encoded) * 8)
            
            if (chunk_idx + 1) % 100 == 0:
                print(f"Processed {chunk_idx + 1}/{total_chunks} chunks")
    
    print(f"Extraction complete. Total Chunks: {len(extracted_parameters)}")
    average_bitrate = (sum(avgbitrate) / len(avgbitrate)) * (sample_rate / WINDOW_SIZE) / 1000
    print(f"Average kbps is {average_bitrate:.2f} Kbps")
    print(f"Max kbps is {(maxvalue * (sample_rate / WINDOW_SIZE) / 1000):.2f} Kbps")

    # --- Synthesis with Multiprocessing ---
    print(f"Generating audio using {num_processes} processes...")
    
    synth_args = [(idx, encoded_data, sample_rate) 
                  for idx, encoded_data in enumerate(extracted_parameters)]
    
    generator_params = {
        'window_size': WINDOW_SIZE,
        'hop_size': HOP_SIZE
    }
    
    synth_func = partial(synthesize_single_chunk,
                        generator_params=generator_params,
                        packer_params=packer_params)
    
    output_signal_length = audio_float.size + WINDOW_SIZE
    output_signal = np.zeros(output_signal_length, dtype=np.float32)
    
    with Pool(processes=num_processes) as pool:
        results = pool.imap(synth_func, synth_args, chunksize=10)
        
        for chunk_idx, synthesized_chunk in results:
            # Overlap-Add
            start_index = chunk_idx * HOP_SIZE
            end_index = start_index + WINDOW_SIZE
            
            add_length = min(synthesized_chunk.size, output_signal_length - start_index)
            output_signal[start_index:end_index] += synthesized_chunk[:add_length]
            
            if (chunk_idx + 1) % 100 == 0:
                print(f"Synthesized {chunk_idx + 1}/{total_chunks} chunks")

    # Normalize and save
    print("Normalizing and saving output...")
    max_abs_val = np.max(np.abs(output_signal))
    if max_abs_val > 1.0e-6:
        output_signal = output_signal / max_abs_val

    output_int16 = (output_signal * 32767).astype(np.int16)
    write(output_audio_path, sample_rate, output_int16)

    print(f"Synthesis complete. Output saved to {output_audio_path}")

if __name__ == '__main__':
    INPUT_FILE = r"sample.wav"
    OUTPUT_FILE = r"output.phxc.wav"

    ANALYSIS_PARAMS = {
        'window_size': 4096,
        'hop_size': 512,
        'min_f0_freq': 20.0,
        'max_f0_freq': 16000.0,
        'peak_threshold': 0.0,
        'max_harmonics_per_f0': 10,
        'max_harmonic_freq_output': 16000.0,
        'max_harmonic_freq_object': 10,
    }

    # Use all available CPU cores (or specify a number)
    process_chunk_by_chunk_multiprocessing(INPUT_FILE, OUTPUT_FILE, ANALYSIS_PARAMS, num_processes=4)

    print("done!")