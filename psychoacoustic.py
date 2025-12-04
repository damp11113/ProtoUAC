import numpy as np


def advanced_psychoacoustic(mdct_coeffs, sample_rate=48000, bands=32,
                            quality=5, ath_offset=0, masking_ratio=1.0,
                            tonality_weight=1.0, spread_factor=0.15, max_freq=None):
    """
    Psychoacoustic masking model inspired by HE-AAC and Opus.

    Args:
        mdct_coeffs: MDCT coefficients (frequency domain)
        sample_rate: Audio sample rate in Hz
        bands: Number of critical bands (typically 32-64)
        quality: Quality level 0-10 (0=max compression, 10=max quality)
                 - Higher = less aggressive masking, preserves more coefficients
        ath_offset: Offset ATH curve in dB (-20 to +20)
                    - Negative = more aggressive (lower threshold, more masking)
                    - Positive = less aggressive (higher threshold, less masking)
        masking_ratio: Global masking strength multiplier (0.1 to 2.0)
                       - Lower = more aggressive masking (more loss)
                       - Higher = less aggressive masking (less loss)
        tonality_weight: Weight for tone vs noise masking (0.0 to 2.0)
                         - 0 = treat everything as noise (more masking)
                         - 2 = emphasize tonal differences
        spread_factor: Spreading function strength (0.0 to 0.5)
                       - Controls how much neighboring bands affect each other
        max_freq: Maximum frequency to keep in Hz (e.g., 4000, 8000, 16000)
                  - None = keep all frequencies (default)
                  - Useful for bandwidth-limited applications (telephone, streaming)

    Returns:
        Masked coefficients with psychoacoustic model applied
    """
    N = len(mdct_coeffs)
    band_size = max(1, N // bands)
    masked = mdct_coeffs.copy()

    # Quality-based adjustments
    # Quality 0-10 maps to masking offsets
    quality = np.clip(quality, 0, 10)
    quality_factor = quality / 10.0

    # Quality affects base masking threshold
    # Lower quality = more aggressive masking
    base_masking_adjust = -10 + (quality_factor * 15)  # -10 dB to +5 dB

    # Clamp parameters
    masking_ratio = np.clip(masking_ratio, 0.1, 2.0)
    tonality_weight = np.clip(tonality_weight, 0.0, 2.0)
    spread_factor = np.clip(spread_factor, 0.0, 0.5)
    ath_offset = np.clip(ath_offset, -20, 20)

    # Bark scale critical band mapping (approximation)
    freqs = np.fft.rfftfreq(N * 2, 1.0 / sample_rate)[:N]

    # Apply max frequency cutoff if specified
    if max_freq is not None:
        max_freq = min(max_freq, sample_rate / 2)  # Can't exceed Nyquist
        cutoff_idx = np.searchsorted(freqs, max_freq)
        if cutoff_idx < N:
            masked[cutoff_idx:] = 0  # Zero out everything above max_freq

    for b in range(bands):
        start = b * band_size
        end = min(start + band_size, N)

        if start >= N:
            break

        # Skip bands above max_freq if specified
        if max_freq is not None and freqs[start] > max_freq:
            masked[start:end] = 0
            continue

        band = masked[start:end]
        band_freqs = freqs[start:end]

        # Calculate band energy (SPL approximation)
        energy = np.sum(band ** 2)

        if energy < 1e-10:
            masked[start:end] = 0
            continue

        # Frequency-dependent absolute threshold of hearing (ATH)
        # Simplified ATH curve in dB SPL
        avg_freq = np.mean(band_freqs) if len(band_freqs) > 0 else 1000
        ath_db = calculate_ath(avg_freq) + ath_offset

        # Convert energy to dB-like scale
        energy_db = 10 * np.log10(energy + 1e-10)

        # Masking threshold calculation
        # 1. Tone masking: -6 dB below masker for nearby frequencies
        # 2. Noise masking: -28 dB below masker
        tone_offset = 6.0  # dB
        noise_offset = 28.0  # dB

        # Calculate tonality measure (spectral flatness)
        geometric_mean = np.exp(np.mean(np.log(np.abs(band) + 1e-10)))
        arithmetic_mean = np.mean(np.abs(band))
        tonality = geometric_mean / (arithmetic_mean + 1e-10)

        # Apply tonality weight
        tonality = np.clip(tonality * tonality_weight, 0, 1)

        # Interpolate between tone and noise masking
        masking_offset = tone_offset + (noise_offset - tone_offset) * (1 - tonality)

        # Apply quality adjustment and masking ratio
        masking_offset = masking_offset - base_masking_adjust
        masking_offset = masking_offset / masking_ratio

        # Calculate masking threshold
        masking_threshold_db = energy_db - masking_offset

        # Apply absolute threshold of hearing
        final_threshold_db = max(masking_threshold_db, ath_db)

        # Convert back to linear scale
        threshold_linear = 10 ** (final_threshold_db / 20.0) * np.sqrt(energy)
        threshold_linear = max(threshold_linear, 1e-8)

        # Apply spreading function (simple neighboring band influence)
        if b > 0 and spread_factor > 0:
            prev_energy = np.sum(masked[max(0, start - band_size):start] ** 2)
            threshold_linear += spread_factor * np.sqrt(prev_energy)

        if b < bands - 1 and spread_factor > 0:
            next_start = min(end, N)
            next_end = min(next_start + band_size, N)
            if next_start < next_end:
                next_energy = np.sum(masked[next_start:next_end] ** 2)
                threshold_linear += spread_factor * np.sqrt(next_energy)

        # Apply masking threshold
        mask = np.abs(band) < threshold_linear / np.sqrt(len(band))
        band[mask] = 0

        masked[start:end] = band

    return masked


def calculate_ath(freq_hz):
    """
    Absolute Threshold of Hearing (ATH) in dB SPL.
    Simplified model based on ISO 226 standard.
    """
    f = freq_hz / 1000.0  # Convert to kHz

    # Simplified ATH formula
    ath = 3.64 * (f ** -0.8) - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + \
          1e-3 * (f ** 4)

    # Normalize to reasonable range
    ath = np.clip(ath, -20, 96)

    return ath


# Example usage with comparison
if __name__ == "__main__":
    # Generate test signal (mix of tones and noise)
    N = 1024
    t = np.linspace(0, 1, N)

    # Simulate MDCT coefficients
    signal = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
    signal += np.sin(2 * np.pi * 880 * t) * 0.3  # 880 Hz tone
    signal += np.random.randn(N) * 0.05  # Noise floor

    # Fake MDCT (just FFT for demo)
    mdct_coeffs = np.fft.rfft(signal)[:N]

    print("=" * 60)
    print("PSYCHOACOUSTIC MASKING COMPARISON")
    print("=" * 60)

    # Test different quality levels
    configs = [
        ("Max Compression", {"quality": 0, "masking_ratio": 0.5}),
        ("Low Quality", {"quality": 2, "masking_ratio": 0.8}),
        ("Medium Quality", {"quality": 5, "masking_ratio": 1.0}),
        ("High Quality", {"quality": 8, "masking_ratio": 1.2}),
        ("Maximum Quality", {"quality": 10, "masking_ratio": 1.5}),
    ]

    original_nonzero = np.count_nonzero(np.abs(mdct_coeffs) > 1e-10)

    for name, params in configs:
        masked_coeffs = advanced_psychoacoustic(mdct_coeffs, sample_rate=44100,
                                                bands=32, **params)

        masked_nonzero = np.count_nonzero(np.abs(masked_coeffs) > 1e-10)
        compression = original_nonzero / max(masked_nonzero, 1)
        zeroed_pct = 100 * (1 - masked_nonzero / original_nonzero)

        print(f"\n{name}:")
        print(f"  Quality: {params['quality']}, Masking Ratio: {params['masking_ratio']}")
        print(f"  Coefficients kept: {masked_nonzero}/{original_nonzero}")
        print(f"  Coefficients zeroed: {zeroed_pct:.1f}%")
        print(f"  Compression ratio: {compression:.2f}x")

    # Test frequency bandwidth limiting
    print("\n" + "=" * 60)
    print("BANDWIDTH LIMITING COMPARISON")
    print("=" * 60)

    bandwidth_configs = [
        ("Full bandwidth (20kHz)", None),
        ("Wideband (16kHz)", 16000),
        ("Super-wideband (12kHz)", 12000),
        ("Wideband phone (8kHz)", 8000),
        ("Narrowband phone (4kHz)", 4000),
    ]

    for name, max_freq in bandwidth_configs:
        masked_coeffs = advanced_psychoacoustic(mdct_coeffs, sample_rate=44100,
                                                bands=32, quality=5, max_freq=max_freq)

        masked_nonzero = np.count_nonzero(np.abs(masked_coeffs) > 1e-10)
        compression = original_nonzero / max(masked_nonzero, 1)
        zeroed_pct = 100 * (1 - masked_nonzero / original_nonzero)

        freq_str = f"{max_freq} Hz" if max_freq else "No limit"
        print(f"\n{name} (max_freq={freq_str}):")
        print(f"  Coefficients kept: {masked_nonzero}/{original_nonzero}")
        print(f"  Coefficients zeroed: {zeroed_pct:.1f}%")
        print(f"  Compression ratio: {compression:.2f}x")

    print("\n" + "=" * 60)
    print("PARAMETER GUIDE:")
    print("=" * 60)
    print("quality (0-10): Higher = better quality, less compression")
    print("masking_ratio (0.1-2.0): Lower = more aggressive masking")
    print("ath_offset (-20 to +20 dB): Negative = more masking")
    print("tonality_weight (0-2): How much to consider tone vs noise")
    print("spread_factor (0-0.5): Neighboring band masking influence")
    print("max_freq (Hz): Limit bandwidth (4000=phone, 8000=wideband, None=full)")
    print("\nCommon max_freq values:")
    print("  - 4000 Hz: Narrowband (telephone quality)")
    print("  - 8000 Hz: Wideband (VoIP, better phone)")
    print("  - 12000 Hz: Super-wideband (high-quality voice)")
    print("  - 16000 Hz: Full-band (near CD quality)")
    print("  - None: No limit (full frequency range)")