import struct
import wave
import numpy as np
import pyaudio
from SubPS import SubPSEncoderv2 as encoder, SubPSDecoderv2 as decoder

CHUNK_SIZE = 1024  # Make this match the frame_size in SubbandEncoder/Decoder
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000

def process_audio():
    p = pyaudio.PyAudio()

    # Initialize encoder and decoder with matching frame size
    subps_enc = encoder(sample_rate=RATE, num_bands=10, frame_size=CHUNK_SIZE // 2, f_min=1000, f_max=8000,
                            resolution="int16")
    subps_dec = decoder(sample_rate=RATE, num_bands=10, frame_size=CHUNK_SIZE // 2, f_min=1000, f_max=8000, resolution="int16", smoothing_factor=10)

    # Open the output stream
    output_stream = p.open(format=SAMPLE_FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           output=True)

    try:
        with wave.open('STD_TEST/JustTheWayYouAre_01.wav', 'rb') as wav_file:
            # Basic file checks
            if wav_file.getnchannels() != 2:
                print("This is not a stereo file.")
                return


            # Process audio frame by frame
            while True:
                # Read a chunk of audio
                frame_data = wav_file.readframes(CHUNK_SIZE)
                if not frame_data:
                    break

                # Convert to numpy array and extract left channel
                frame_block = np.frombuffer(frame_data, dtype=np.int16)
                frame_block = frame_block.reshape(-1, 2)  # Reshape for stereo


                # process is here
                mono_output, para_output = subps_enc.encode(frame_block.astype(np.int16))


                print(f"para_output: {len(para_output)}")

                processed_data = subps_dec.decode(mono_output, para_output)

                # Write to output stream
                output_stream.write(np.clip(frame_block, -32768, 32767).tobytes())

    finally:
        # Clean up
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()


if __name__ == "__main__":
    process_audio()