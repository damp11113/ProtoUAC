import struct
import wave
import numpy as np
import pyaudio
from Subband import SubbandEncoder, SubbandDecoder

CHUNK_SIZE = 1024  # Make this match the frame_size in SubbandEncoder/Decoder
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

def process_audio():
    p = pyaudio.PyAudio()

    # Initialize encoder and decoder with matching frame size
    sbcenc = SubbandEncoder(sample_rate=RATE, num_bands=32, frame_size=CHUNK_SIZE // 2, f_min=4000, f_max=16000,
                            resolution="int8")
    sbcdec = SubbandDecoder(sample_rate=RATE, num_bands=32, frame_size=CHUNK_SIZE // 2, f_min=4000, f_max=16000,
                            resolution="int8")

    # Open the output stream
    output_stream = p.open(format=SAMPLE_FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           output=True)

    try:
        with wave.open('input.wav', 'rb') as wav_file:
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
                left_channel = frame_block[:, 0]

                # Skip if we don't have a full frame
                if len(left_channel) < CHUNK_SIZE:
                    break

                # Encode the frame
                encoded_frame = sbcenc.encode(left_channel)

                print(len(encoded_frame))

                # Decode the frame
                decoded_frame = sbcdec.decode(encoded_frame)

                # Write to output stream
                output_stream.write(np.clip(decoded_frame, -32768, 32767).tobytes())

    finally:
        # Clean up
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()


if __name__ == "__main__":
    process_audio()