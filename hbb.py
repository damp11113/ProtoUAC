# Hybrid Bandbands coding
# Use Harmonic + SBR Noise

import numpy as np
from harmonic_extraction import hps
from phxc import HarmonicExtractor, HarmonicGenerator
from sbr import SBREncoder, SBRDecoder

class HyBBEncoder:
    def __init__(
        self,
        sample_rate,
        frame_size=2048,
        hop_size=1024,
        min_freq=20.0,
        max_freq=14000.0,
        min_freq_imp=250.0,
        max_freq_imp=20000,
        imp_point=32,
        max_harmonics_per_f0=10,
        max_harmonic_freq_object=10,
        log_scale=True,
        hps_L_h=0.1,
        hps_L_p=1000
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.hps_L_h = hps_L_h
        self.hps_L_p = hps_L_p
        
        # Initialize harmonic encoder
        self.harmonic_encoder = HarmonicExtractor(
            sample_rate=sample_rate,
            window_size=frame_size,
            hop_size=hop_size,
            min_f0_freq=min_freq,
            max_f0_freq=max_freq,
            peak_threshold=0.0,
            max_harmonics_per_f0=max_harmonics_per_f0,
            max_harmonic_freq_output=max_freq,
            max_harmonic_freq_object=max_harmonic_freq_object,
            log=log_scale
        )
        
        # Initialize percussive encoder
        self.impulse_encoder = SBREncoder(
            sample_rate,
            min_freq_imp,
            max_freq_imp,
            imp_point,
            -80,
            forcing=1 # Noise only
        )

    def encode(self, mono_signal):
        # Separate harmonic and percussive components
        harmonic_audio, percussive_audio = hps(
            mono_signal,
            Fs=self.sample_rate,
            N=self.frame_size,
            H=self.hop_size,
            L_h=self.hps_L_h,
            L_p=self.hps_L_p
        )
        
        # Encode harmonic signal stereo parameters
        harmonic_sd = self.harmonic_encoder.process_chunk(harmonic_audio)
        
        # Encode percussive signal stereo parameters
        impulse_sd, shouldUseNoises, is_transient = self.impulse_encoder.analyze(percussive_audio, True)
        
        return harmonic_sd, impulse_sd

class HyBBDecoder:    
    def __init__(
        self,
        sample_rate,
        frame_size=2048,
        hop_size=1024,
        min_freq_imp=250.0,
        max_freq_imp=20000,
        imp_point=32
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        
        # Initialize harmonic decoder
        self.harmonic_decoder = HarmonicGenerator(
            sample_rate=sample_rate,
            window_size=frame_size,
            hop_size=hop_size
        )
        
        # Initialize percussive decoder
        self.impulse_decoder = SBRDecoder(
            sample_rate,
            min_freq_imp,
            max_freq_imp,
            imp_point,
            chunk_size=frame_size
        )

    def decode(self, harmonic_sd, impulse_sd):        
        # Reconstruct harmonic
        ho = self.harmonic_decoder.process_chunk(harmonic_sd)
        
        #print(impulse_sd)
        # Reconstruct percussive
        imo = self.impulse_decoder.generate(None, impulse_sd, False, True, True)
        
        # Ensure both components have the same length
        min_len = min(
            ho.shape[0],
            imo.shape[0]
        )
        
        # Mix components
        reconstructed = (
            ho[:min_len] +
            imo[:min_len] * 0.25
        )
        
        return reconstructed