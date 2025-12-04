import math
import numpy as np
import struct
from typing import List, Optional, Tuple
from enum import IntEnum

class CompressionMode(IntEnum):
    FLOAT32 = 0
    INT16 = 1
    INT8 = 2


class SBRDataPacker:
    """Pack SBR band energies with delta encoding and compression."""

    def __init__(
            self,
            min_db: float = -75.0,
            max_db: float = 0.0,
            delta_threshold: float = 0.5
    ):
        """
        Initialize packer with expected dB range.

        Args:
            min_db: Minimum expected dB value (floor)
            max_db: Maximum expected dB value
            delta_threshold: Minimum change in dB to encode (higher = more compression)
        """
        self.min_db = min_db
        self.max_db = max_db
        self.db_range = max_db - min_db
        self.delta_threshold = delta_threshold

        # Store previous frame for delta encoding
        self.prev_energies: Optional[List[float]] = None
        self.prev_noises: Optional[List[bool]] = None

    def pack(
            self,
            band_energies: List[float],
            is_transient: bool,
            shouldUseNoises: Optional[List[bool]] = None,
            mode: CompressionMode = CompressionMode.FLOAT32,
            use_delta: bool = True
    ) -> bytes:
        """
        Pack band energies with delta encoding and noise flags.

        Args:
            band_energies: List of band energy values in dB
            is_transient: Transient detection flag
            shouldUseNoises: Optional list of noise flags per band (True = use noise)
            mode: Compression mode (FLOAT32, INT16, or INT8)
            use_delta: Enable delta encoding (only send changed values)

        Returns:
            Packed bytes with header and data
        """
        num_bands = len(band_energies)

        # Validate shouldUseNoises if provided
        if shouldUseNoises is not None and len(shouldUseNoises) != num_bands:
            raise ValueError(f"shouldUseNoises length ({len(shouldUseNoises)}) must match band_energies ({num_bands})")

        # Force full frame on transients or if no previous data
        if is_transient or self.prev_energies is None or not use_delta:
            is_delta = False
            values_to_encode = band_energies
            changed_indices = list(range(num_bands))
        else:
            # Delta encoding: only send changed values
            is_delta = True
            changed_indices = []
            values_to_encode = []

            for i, (curr, prev) in enumerate(zip(band_energies, self.prev_energies)):
                # Check if energy changed
                energy_changed = abs(curr - prev) >= self.delta_threshold

                # Check if noise flag changed (if using noise flags)
                noise_changed = False
                if shouldUseNoises is not None and self.prev_noises is not None:
                    noise_changed = shouldUseNoises[i] != self.prev_noises[i]

                # Include if either changed
                if energy_changed or noise_changed:
                    changed_indices.append(i)
                    values_to_encode.append(curr)

        # Store for next frame
        self.prev_energies = band_energies.copy()
        self.prev_noises = shouldUseNoises.copy() if shouldUseNoises is not None else None

        # Header: version(1) + flags(1) + mode(1) + num_bands(2) + num_changed(2)
        version = 1
        has_noise_flags = shouldUseNoises is not None
        flags = (int(is_transient) << 0) | (int(is_delta) << 1) | (int(has_noise_flags) << 2)
        num_changed = len(changed_indices)

        header = struct.pack(
            '=BBBHH',
            version,
            flags,
            mode,
            num_bands,
            num_changed
        )

        # Pack changed indices (only if delta encoding)
        if is_delta and num_changed > 0:
            # Use 1 byte per index if num_bands <= 255, else 2 bytes
            if num_bands <= 255:
                indices_data = struct.pack(f'{num_changed}B', *changed_indices)
            else:
                indices_data = struct.pack(f'{num_changed}H', *changed_indices)
        else:
            indices_data = b''

        # Pack values based on compression mode
        if mode == CompressionMode.FLOAT32:
            values_data = struct.pack(f'{num_changed}f', *values_to_encode)

        elif mode == CompressionMode.INT16:
            normalized = [
                (energy - self.min_db) / self.db_range
                for energy in values_to_encode
            ]
            int_values = [
                int(max(-32768, min(32767, n * 65535 - 32768)))
                for n in normalized
            ]
            values_data = struct.pack(f'{num_changed}h', *int_values)

        elif mode == CompressionMode.INT8:
            normalized = [
                (energy - self.min_db) / self.db_range
                for energy in values_to_encode
            ]
            int_values = [
                int(max(-128, min(127, n * 255 - 128)))
                for n in normalized
            ]
            values_data = struct.pack(f'{num_changed}b', *int_values)

        else:
            raise ValueError(f"Unsupported compression mode: {mode}")

        # Pack noise flags if present
        if shouldUseNoises is not None:
            if is_delta:
                # Only pack flags for changed indices
                noise_flags = [shouldUseNoises[i] for i in changed_indices]
            else:
                # Pack all flags
                noise_flags = shouldUseNoises

            # Pack as bits (8 flags per byte)
            noise_data = self._pack_bool_flags(noise_flags)
        else:
            noise_data = b''

        return header + indices_data + values_data + noise_data

    def _pack_bool_flags(self, flags: List[bool]) -> bytes:
        """Pack boolean flags into bytes (8 flags per byte)."""
        num_bytes = (len(flags) + 7) // 8
        result = bytearray(num_bytes)

        for i, flag in enumerate(flags):
            if flag:
                byte_idx = i // 8
                bit_idx = i % 8
                result[byte_idx] |= (1 << bit_idx)

        return bytes(result)

    def _unpack_bool_flags(self, data: bytes, num_flags: int) -> List[bool]:
        """Unpack boolean flags from bytes."""
        flags = []
        for i in range(num_flags):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(data):
                flags.append(bool(data[byte_idx] & (1 << bit_idx)))
            else:
                flags.append(False)
        return flags

    def reset(self):
        """Reset encoder state (call when starting new stream)."""
        self.prev_energies = None
        self.prev_noises = None

    def get_packed_size(
            self,
            num_bands: int,
            num_changed: int,
            mode: CompressionMode,
            has_noise_flags: bool = False
    ) -> int:
        """Get the size of packed data in bytes."""
        header_size = 7

        # Index size
        if num_changed > 0:
            idx_size = num_changed if num_bands <= 255 else num_changed * 2
        else:
            idx_size = 0

        # Value size
        if mode == CompressionMode.FLOAT32:
            val_size = num_changed * 4
        elif mode == CompressionMode.INT16:
            val_size = num_changed * 2
        elif mode == CompressionMode.INT8:
            val_size = num_changed * 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Noise flags size (if present)
        noise_size = 0
        if has_noise_flags:
            noise_size = (num_changed + 7) // 8  # Ceiling division for bit packing

        return header_size + idx_size + val_size + noise_size

    def get_compression_ratio(
            self,
            num_bands: int,
            num_changed: int,
            mode: CompressionMode,
            has_noise_flags: bool = False
    ) -> float:
        """Calculate compression ratio vs full FLOAT32 frame."""
        full_size = 7 + num_bands * 4
        if has_noise_flags:
            full_size += (num_bands + 7) // 8

        packed_size = self.get_packed_size(num_bands, num_changed, mode, has_noise_flags)
        return full_size / packed_size


class SBRDataUnpacker:
    """Unpack SBR band energies with delta decoding and optional interpolation."""

    def __init__(
            self,
            min_db: float = -75.0,
            max_db: float = 0.0,
            use_interpolation: bool = True
    ):
        """
        Initialize unpacker with expected dB range.

        Args:
            min_db: Minimum expected dB value (floor)
            max_db: Maximum expected dB value
            use_interpolation: Enable interpolation for missing values
        """
        self.min_db = min_db
        self.max_db = max_db
        self.db_range = max_db - min_db
        self.use_interpolation = use_interpolation

    def unpack(
            self,
            data: bytes,
            prev_frame: Optional[List[float]] = None,
            prev_noises: Optional[List[bool]] = None
    ) -> Tuple[List[float], bool, CompressionMode, Optional[List[bool]]]:
        """
        Unpack bytes back to band energies with delta decoding.

        Args:
            data: Packed bytes
            prev_frame: Previous frame data (required for delta frames)
            prev_noises: Previous noise flags (required for delta frames with noise)

        Returns:
            Tuple of (band_energies, is_transient, mode, shouldUseNoises)
        """
        if len(data) < 7:
            raise ValueError("Data too short, invalid format")

        # Unpack header
        version, flags, mode_val, num_bands, num_changed = struct.unpack(
            '=BBBHH',
            data[:7]
        )

        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        mode = CompressionMode(mode_val)
        is_transient = bool(flags & 0x01)
        is_delta = bool(flags & 0x02)
        has_noise_flags = bool(flags & 0x04)

        offset = 7

        # Read changed indices
        if is_delta and num_changed > 0:
            if num_bands <= 255:
                idx_size = num_changed
                changed_indices = list(struct.unpack(
                    f'{num_changed}B',
                    data[offset:offset + idx_size]
                ))
            else:
                idx_size = num_changed * 2
                changed_indices = list(struct.unpack(
                    f'{num_changed}H',
                    data[offset:offset + idx_size]
                ))
            offset += idx_size
        else:
            changed_indices = list(range(num_bands))

        # Read values
        if mode == CompressionMode.FLOAT32:
            val_size = num_changed * 4
        elif mode == CompressionMode.INT16:
            val_size = num_changed * 2
        elif mode == CompressionMode.INT8:
            val_size = num_changed
        else:
            raise ValueError(f"Unsupported compression mode: {mode}")

        payload = data[offset:offset + val_size]
        offset += val_size

        if mode == CompressionMode.FLOAT32:
            if len(payload) != val_size:
                raise ValueError(f"Expected {val_size} bytes, got {len(payload)}")
            changed_values = list(struct.unpack(f'{num_changed}f', payload))

        elif mode == CompressionMode.INT16:
            if len(payload) != val_size:
                raise ValueError(f"Expected {val_size} bytes, got {len(payload)}")
            int_values = struct.unpack(f'{num_changed}h', payload)
            changed_values = [
                (val + 32768) / 65535 * self.db_range + self.min_db
                for val in int_values
            ]

        elif mode == CompressionMode.INT8:
            if len(payload) != val_size:
                raise ValueError(f"Expected {val_size} bytes, got {len(payload)}")
            int_values = struct.unpack(f'{num_changed}b', payload)
            changed_values = [
                (val + 128) / 255 * self.db_range + self.min_db
                for val in int_values
            ]

        # Read noise flags if present
        shouldUseNoises = None
        if has_noise_flags:
            noise_bytes = (num_changed + 7) // 8
            noise_data = data[offset:offset + noise_bytes]
            changed_noise_flags = self._unpack_bool_flags(noise_data, num_changed)

            # Reconstruct full noise array if delta
            if is_delta:
                if prev_noises is None:
                    raise ValueError("Delta frame with noise flags requires prev_noises")
                if len(prev_noises) != num_bands:
                    raise ValueError(f"Previous noise size mismatch: expected {num_bands}, got {len(prev_noises)}")

                shouldUseNoises = prev_noises.copy()
                for idx, flag in zip(changed_indices, changed_noise_flags):
                    shouldUseNoises[idx] = flag
            else:
                shouldUseNoises = changed_noise_flags

        # Reconstruct full frame
        if is_delta:
            if prev_frame is None:
                raise ValueError("Delta frame requires previous frame data")
            if len(prev_frame) != num_bands:
                raise ValueError(f"Previous frame size mismatch: expected {num_bands}, got {len(prev_frame)}")

            # Start with previous frame
            band_energies = prev_frame.copy()

            # Update changed values
            for idx, val in zip(changed_indices, changed_values):
                band_energies[idx] = val

            # Optional: Interpolate unchanged values for smoothness
            if self.use_interpolation and len(changed_indices) < num_bands:
                band_energies = self._interpolate_missing(
                    band_energies,
                    changed_indices,
                    num_bands
                )
        else:
            # Full frame
            band_energies = changed_values

        return band_energies, is_transient, mode, shouldUseNoises

    def _unpack_bool_flags(self, data: bytes, num_flags: int) -> List[bool]:
        """Unpack boolean flags from bytes."""
        flags = []
        for i in range(num_flags):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(data):
                flags.append(bool(data[byte_idx] & (1 << bit_idx)))
            else:
                flags.append(False)
        return flags

    def _interpolate_missing(
            self,
            values: List[float],
            changed_indices: List[int],
            num_bands: int
    ) -> List[float]:
        """
        Interpolate unchanged values between changed values for smoothness.
        """
        if len(changed_indices) <= 1:
            return values

        result = values.copy()
        changed_set = set(changed_indices)

        for i in range(num_bands):
            if i in changed_set:
                continue

            # Find nearest changed indices before and after
            prev_idx = None
            next_idx = None

            for idx in changed_indices:
                if idx < i:
                    prev_idx = idx
                elif idx > i and next_idx is None:
                    next_idx = idx
                    break

            # Interpolate between neighbors
            if prev_idx is not None and next_idx is not None:
                # Linear interpolation
                t = (i - prev_idx) / (next_idx - prev_idx)
                result[i] = values[prev_idx] * (1 - t) + values[next_idx] * t
            elif prev_idx is not None:
                # Extrapolate from previous
                result[i] = values[prev_idx] * 0.7 + result[i] * 0.3
            elif next_idx is not None:
                # Extrapolate from next
                result[i] = values[next_idx] * 0.7 + result[i] * 0.3

        return result

def pack_stereo_metadata(pan_values, ipd_values, ic_values, min_freq, max_freq, point, n_bands):
    """
    Pack PS metadata:
    - pan_values: list of floats [-1, 1]
    - ipd_values: list of floats [-pi, pi]
    - ic_values: list of bools
    - min_freq: int (minimum frequency)
    - max_freq: int (maximum frequency)
    - point: int
    - n_bands: int (number of bands)
    Returns: bytes
    """
    n = len(pan_values)
    if not (len(ipd_values) == len(ic_values) == n):
        raise ValueError("All input lists must have same length")

    if n != n_bands:
        raise ValueError(f"n_bands ({n_bands}) must match length of input lists ({n})")

    packed_bytes = bytearray()

    # Pack 4 integers at the beginning (4 bytes each = 16 bytes total)
    packed_bytes.extend(struct.pack('<i', min_freq))
    packed_bytes.extend(struct.pack('<i', max_freq))
    packed_bytes.extend(struct.pack('<i', point))
    packed_bytes.extend(struct.pack('<i', n_bands))

    # Pack PAN and IPD as int8
    for pan, ipd in zip(pan_values, ipd_values):
        pan_byte = int(round(pan * 127))
        ipd_byte = int(round(ipd / math.pi * 127))
        pan_byte = max(-128, min(127, pan_byte))
        ipd_byte = max(-128, min(127, ipd_byte))
        packed_bytes.append(pan_byte & 0xFF)
        packed_bytes.append(ipd_byte & 0xFF)

    # Pack IC as bits (1 bit per band)
    ic_byte = 0
    bit_count = 0
    for ic in ic_values:
        ic_byte = (ic_byte << 1) | (1 if ic else 0)
        bit_count += 1
        if bit_count == 8:
            packed_bytes.append(ic_byte & 0xFF)
            ic_byte = 0
            bit_count = 0

    # Remaining bits
    if bit_count > 0:
        ic_byte = ic_byte << (8 - bit_count)
        packed_bytes.append(ic_byte & 0xFF)

    return bytes(packed_bytes)


def unpack_stereo_metadata(packed_bytes):
    """
    Unpack PS metadata
    Returns: (pan_values, ipd_values, ic_values, min_freq, max_freq, point, n_bands)
    """
    # Unpack 4 integers from the beginning
    min_freq = struct.unpack('<i', packed_bytes[0:4])[0]
    max_freq = struct.unpack('<i', packed_bytes[4:8])[0]
    point = struct.unpack('<i', packed_bytes[8:12])[0]
    n_bands = struct.unpack('<i', packed_bytes[12:16])[0]

    pan_values = []
    ipd_values = []
    ic_values = []

    # PAN/IPD start after the header (16 bytes)
    header_size = 16
    for i in range(n_bands):
        offset = header_size + i * 2
        pan_byte = struct.unpack('b', packed_bytes[offset:offset + 1])[0]
        ipd_byte = struct.unpack('b', packed_bytes[offset + 1:offset + 2])[0]
        pan = pan_byte / 127.0
        ipd = ipd_byte / 127.0 * math.pi
        pan_values.append(pan)
        ipd_values.append(ipd)

    # IC bits start after PAN/IPD
    ic_start = header_size + n_bands * 2
    total_ic_bits = n_bands
    bits_read = 0

    for b in packed_bytes[ic_start:]:
        for i in range(7, -1, -1):
            if bits_read >= total_ic_bits:
                break
            bit = (b >> i) & 1
            ic_values.append(bool(bit))
            bits_read += 1

    return pan_values, ipd_values, ic_values, min_freq, max_freq, point


def quantize_coefficients(coeffs, num_bits=8):
    """
    Simple uniform scalar quantization

    Parameters:
    -----------
    coeffs : np.array
        MDCT coefficients to quantize
    num_bits : int
        Number of bits for quantization (default: 8)

    Returns:
    --------
    quantized : np.array (int)
        Quantized coefficients
    scale : float
        Scale factor for dequantization
    """
    if len(coeffs) == 0:
        return np.array([]), 0.0

    # Find max absolute value for scaling
    max_val = np.max(np.abs(coeffs))

    if max_val == 0:
        return np.zeros_like(coeffs, dtype=np.int16), 0.0

    # Calculate quantization levels
    max_level = (2 ** (num_bits - 1)) - 1  # Leave one bit for sign
    scale = max_val / max_level

    # Quantize
    quantized = np.round(coeffs / scale).astype(np.int16)

    return quantized, scale


def dequantize_coefficients(quantized, scale):
    """
    Dequantize coefficients

    Parameters:
    -----------
    quantized : np.array (int)
        Quantized coefficients
    scale : float
        Scale factor from quantization

    Returns:
    --------
    coeffs : np.array (float)
        Dequantized coefficients
    """
    return quantized.astype(np.float32) * scale

class HarmonicPacker:
    """
    Enhanced harmonic packer with multiple encoding options.

    Data types:
    - 'uint8': 0-255 (1 byte per value)
    - 'uint16': 0-65535 (2 bytes per value)
    - 'int8': -128 to 127 (1 byte per value)
    - 'int16': -32768 to 32767 (2 bytes per value)
    - 'float16': half precision (2 bytes per value)
    - 'float32': single precision (4 bytes per value)

    Scaling modes:
    - 'linear': Direct linear scaling
    - 'log': Logarithmic scaling (better for audio amplitudes)
    - 'db': Decibel scaling (perceptually accurate)
    """

    def __init__(self, sample_rate, window_size,
                 amp_dtype='uint8', phase_dtype='uint8',
                 scale_mode='log', min_amp_db=-80):
        self.sr = sample_rate
        self.win = window_size
        self.bin_size = sample_rate / window_size
        self.amp_dtype = amp_dtype
        self.phase_dtype = phase_dtype
        self.scale_mode = scale_mode
        self.min_amp_db = min_amp_db
        self.min_amp_linear = 10 ** (min_amp_db / 20)

        # Define packing formats
        self.dtype_formats = {
            'uint8': 'B',
            'uint16': 'H',
            'int8': 'b',
            'int16': 'h',
            'float16': None,  # special handling
            'float32': 'f'
        }

        # Define value ranges
        self.dtype_ranges = {
            'uint8': (0, 255),
            'uint16': (0, 65535),
            'int8': (-128, 127),
            'int16': (-32768, 32767),
            'float16': (None, None),  # no quantization needed
            'float32': (None, None)  # no quantization needed
        }

    def _encode_amplitude(self, amp, max_amp):
        """Encode amplitude based on selected data type and scale mode."""
        if self.amp_dtype in ['float16', 'float32']:
            # Direct float storage, no quantization
            return amp

        # Normalize
        if max_amp < self.min_amp_linear:
            max_amp = self.min_amp_linear

        if self.scale_mode == 'linear':
            norm = amp / max_amp
        elif self.scale_mode == 'log':
            # Log scale: better resolution at low amplitudes
            amp_safe = max(amp, self.min_amp_linear)
            max_safe = max(max_amp, self.min_amp_linear)
            try:
                norm = math.log(amp_safe / self.min_amp_linear) / math.log(max_safe / self.min_amp_linear)
            except ZeroDivisionError:
                norm = 0.0

            norm = np.clip(norm, 0, 1)
        elif self.scale_mode == 'db':
            # Decibel scale
            amp_safe = max(amp, self.min_amp_linear)
            max_safe = max(max_amp, self.min_amp_linear)
            amp_db = 20 * math.log10(amp_safe)
            max_db = 20 * math.log10(max_safe)
            norm = (amp_db - self.min_amp_db) / (max_db - self.min_amp_db)
            norm = np.clip(norm, 0, 1)
        else:
            norm = amp / max_amp

        # Quantize to integer range
        vmin, vmax = self.dtype_ranges[self.amp_dtype]
        return int(norm * (vmax - vmin) + vmin)

    def _decode_amplitude(self, encoded_val, max_amp):
        """Decode amplitude from encoded value."""
        if self.amp_dtype in ['float16', 'float32']:
            return encoded_val

        vmin, vmax = self.dtype_ranges[self.amp_dtype]
        norm = (encoded_val - vmin) / (vmax - vmin)

        if self.scale_mode == 'linear':
            return norm * max_amp
        elif self.scale_mode == 'log':
            max_safe = max(max_amp, self.min_amp_linear)
            log_ratio = math.log(max_safe / self.min_amp_linear)
            return self.min_amp_linear * math.exp(norm * log_ratio)
        elif self.scale_mode == 'db':
            max_safe = max(max_amp, self.min_amp_linear)
            max_db = 20 * math.log10(max_safe)
            amp_db = norm * (max_db - self.min_amp_db) + self.min_amp_db
            return 10 ** (amp_db / 20)
        else:
            return norm * max_amp

    def _encode_phase(self, phase):
        """Encode phase to selected data type."""
        if self.phase_dtype in ['float16', 'float32']:
            return phase

        # Normalize phase from [-π, π] to [0, 1]
        norm = (phase + math.pi) / (2 * math.pi)

        vmin, vmax = self.dtype_ranges[self.phase_dtype]
        return int(norm * (vmax - vmin) + vmin)

    def _decode_phase(self, encoded_val):
        """Decode phase from encoded value."""
        if self.phase_dtype in ['float16', 'float32']:
            return encoded_val

        vmin, vmax = self.dtype_ranges[self.phase_dtype]
        norm = (encoded_val - vmin) / (vmax - vmin)
        return norm * (2 * math.pi) - math.pi

    def _pack_value(self, value, dtype):
        """Pack a single value according to its data type."""
        if dtype == 'float16':
            return np.float16(value).tobytes()
        elif dtype == 'float32':
            return struct.pack('f', value)
        else:
            fmt = self.dtype_formats[dtype]
            return struct.pack(fmt, value)

    def _unpack_value(self, data, pos, dtype):
        """Unpack a single value and return (value, new_position)."""
        if dtype == 'float16':
            value = np.frombuffer(data[pos:pos + 2], dtype=np.float16)[0]
            return float(value), pos + 2
        elif dtype == 'float32':
            value = struct.unpack_from('f', data, pos)[0]
            return value, pos + 4
        else:
            fmt = self.dtype_formats[dtype]
            size = struct.calcsize(fmt)
            value = struct.unpack_from(fmt, data, pos)[0]
            return value, pos + size

    def pack_chunk(self, harmonic_objects):
        """Pack harmonic objects into bytes."""
        payload = bytearray()

        for obj in harmonic_objects:
            freq = obj["freq"]
            harms = obj["harmonics"]
            hcount = len(harms)

            # Convert frequency to bin
            freq_bin = int(freq / self.bin_size)

            # Find max amplitude for scaling
            max_amp_obj = max(h['amp'] for h in harms) if harms else self.min_amp_linear

            # Pack header: freq_bin (uint16), hcount (uint8), scale_factor (float32)
            payload += struct.pack('<HB', freq_bin, hcount)
            payload += struct.pack('f', max_amp_obj)

            # Pack each harmonic
            for h in harms:
                encoded_amp = self._encode_amplitude(h["amp"], max_amp_obj)
                encoded_phase = self._encode_phase(h["phase"])

                payload += self._pack_value(encoded_amp, self.amp_dtype)
                payload += self._pack_value(encoded_phase, self.phase_dtype)

        return bytes(payload)

    def unpack_chunk(self, data):
        """Unpack bytes back into harmonic objects."""
        pos = 0
        length = len(data)
        objects = []

        while pos < length:
            # Read header
            freq_bin, hcount = struct.unpack_from('<HB', data, pos)
            pos += 3

            max_amp_obj = struct.unpack_from('f', data, pos)[0]
            pos += 4

            freq = freq_bin * self.bin_size
            harmonics = []

            # Read harmonics
            for _ in range(hcount):
                encoded_amp, pos = self._unpack_value(data, pos, self.amp_dtype)
                encoded_phase, pos = self._unpack_value(data, pos, self.phase_dtype)

                amp = self._decode_amplitude(encoded_amp, max_amp_obj)
                phase = self._decode_phase(encoded_phase)

                harmonics.append({'amp': amp, 'phase': phase})

            obj = {
                "freq": freq,
                "n_harmonic": hcount,
                "main_amp": harmonics[0]["amp"] if harmonics else 0.0,
                "harmonics": harmonics
            }
            objects.append(obj)

        return objects

    def get_bytes_per_harmonic(self):
        """Calculate bytes per harmonic for current configuration."""
        amp_size = 2 if self.amp_dtype == 'float16' else struct.calcsize(self.dtype_formats.get(self.amp_dtype, 'B'))
        phase_size = 2 if self.phase_dtype == 'float16' else struct.calcsize(
            self.dtype_formats.get(self.phase_dtype, 'B'))
        return amp_size + phase_size