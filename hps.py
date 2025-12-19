import numpy as np
from harmonic_extraction import hps, hps_stereo
from phsc import HarmonicStereoExtractor, HarmonicStereoSynthesizer
from parametric_coding import PSEncoder, PSDecoder

class HyPSEncoder:
    def __init__(
        self,
        sample_rate,
        frame_size=2048,
        hop_size=1024,
        min_freq=120.0,
        max_freq=16000.0,
        imp_point=5,
        max_harmonics_per_f0=5,
        max_harmonic_freq_object=5,
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
        self.harmonic_encoder = HarmonicStereoExtractor(
            sample_rate=sample_rate,
            window_size=frame_size,
            hop_size=hop_size,
            min_f0_freq=min_freq,
            max_f0_freq=max_freq,
            peak_threshold=0.0,
            max_harmonics_per_f0=max_harmonics_per_f0,
            max_harmonic_freq_object=max_harmonic_freq_object,
            log=log_scale
        )
        
        # Initialize percussive encoder
        self.impulse_encoder = PSEncoder(
            sample_rate,
            min_freq,
            max_freq,
            imp_point,
            -50,
            use_grouping=False,
            log_scale=log_scale
        )

    def encode(self, stereo_signal):
        # Separate harmonic and percussive components
        harmonic_audio, percussive_audio = hps_stereo(
            stereo_signal,
            Fs=self.sample_rate,
            N=self.frame_size,
            H=self.hop_size,
            L_h=self.hps_L_h,
            L_p=self.hps_L_p
        )
        
        # Encode harmonic signal stereo parameters
        harmonic_sd = self.harmonic_encoder.process_chunk(
            harmonic_audio[:, 0],
            harmonic_audio[:, 1]
        )
        
        # Encode percussive signal stereo parameters
        impulse_sd = self.impulse_encoder.analyze(percussive_audio, True)
        
        return harmonic_sd, impulse_sd

class HyPSDecoder:    
    def __init__(
        self,
        sample_rate,
        frame_size=2048,
        hop_size=1024,
        min_freq=120.0,
        max_freq=16000.0,
        imp_point=5,
        log_scale=True,
        stereo_width=1.5,
        hps_L_h=0.1,
        hps_L_p=1000
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.hps_L_h = hps_L_h
        self.hps_L_p = hps_L_p
        
        # Initialize harmonic decoder
        self.harmonic_decoder = HarmonicStereoSynthesizer(
            sample_rate=sample_rate,
            window_size=frame_size,
            hop_size=hop_size,
            stereo_width=stereo_width
        )
        
        # Initialize percussive decoder
        self.impulse_decoder = PSDecoder(
            sample_rate,
            min_freq,
            max_freq,
            imp_point,
            use_grouping=True,
            log_scale=log_scale
        )

    def decode(self, mono_audio, harmonic_sd, impulse_sd):
        if mono_audio.ndim != 1:
            raise ValueError("Input must be mono signal")

        # Separate mono signal into harmonic and percussive
        harmonic_audio, percussive_audio = hps(
            mono_audio,
            Fs=self.sample_rate,
            N=self.frame_size,
            H=self.hop_size,
            L_h=self.hps_L_h,
            L_p=self.hps_L_p
        )
        
        # Reconstruct harmonic stereo
        rh_l, rh_r = self.harmonic_decoder.process_chunk(
            harmonic_audio,
            harmonic_sd,
            False
        )

        reconstructed_harmonic_stereo = np.column_stack([rh_l, rh_r])
        
        # Extract spatial parameters for percussive
        pan_values = [pan for freq, pan, ipd, ic in impulse_sd]
        ipd_values = [ipd for freq, pan, ipd, ic in impulse_sd]
        ic_values = [ic >= 1 for freq, pan, ipd, ic in impulse_sd]
        
        # Reconstruct percussive stereo
        reconstructed_impulse_stereo = self.impulse_decoder.apply(
            mono_audio=percussive_audio,
            pan_values=pan_values,
            ipd_values=ipd_values,
            ic_values=ic_values
        )
        
        # Ensure both components have the same length
        min_len = min(
            reconstructed_harmonic_stereo.shape[0],
            reconstructed_impulse_stereo.shape[0]
        )
        
        # Mix components
        reconstructed_stereo = (
            reconstructed_harmonic_stereo[:min_len] +
            reconstructed_impulse_stereo[:min_len]
        )
        
        return reconstructed_stereo