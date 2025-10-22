import math
import struct
from typing import List, Tuple, Optional
from enum import IntEnum

class CompressionMode(IntEnum):
    """Compression modes for band energy data."""
    FLOAT32 = 0  # 4 bytes per value, full precision
    INT16 = 1  # 2 bytes per value, -32768 to 32767 range
    INT8 = 2  # 1 byte per value, -128 to 127 range


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

    def pack(
            self,
            band_energies: List[float],
            is_transient: bool,
            mode: CompressionMode = CompressionMode.FLOAT32,
            use_delta: bool = True
    ) -> bytes:
        """
        Pack band energies with delta encoding.

        Args:
            band_energies: List of band energy values in dB
            is_transient: Transient detection flag
            mode: Compression mode (FLOAT32, INT16, or INT8)
            use_delta: Enable delta encoding (only send changed values)

        Returns:
            Packed bytes with header and data
        """
        num_bands = len(band_energies)

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
                if abs(curr - prev) >= self.delta_threshold:
                    changed_indices.append(i)
                    values_to_encode.append(curr)

        # Store for next frame
        self.prev_energies = band_energies.copy()

        # Header: version(1) + flags(1) + mode(1) + num_bands(2) + num_changed(2)
        version = 1
        flags = (int(is_transient) << 0) | (int(is_delta) << 1)
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

        return header + indices_data + values_data

    def reset(self):
        """Reset encoder state (call when starting new stream)."""
        self.prev_energies = None

    def get_packed_size(self, num_bands: int, num_changed: int, mode: CompressionMode) -> int:
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

        return header_size + idx_size + val_size

    def get_compression_ratio(
            self,
            num_bands: int,
            num_changed: int,
            mode: CompressionMode
    ) -> float:
        """Calculate compression ratio vs full FLOAT32 frame."""
        full_size = 7 + num_bands * 4
        packed_size = self.get_packed_size(num_bands, num_changed, mode)
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
            prev_frame: Optional[List[float]] = None
    ) -> Tuple[List[float], bool, CompressionMode]:
        """
        Unpack bytes back to band energies with delta decoding.

        Args:
            data: Packed bytes
            prev_frame: Previous frame data (required for delta frames)

        Returns:
            Tuple of (band_energies, is_transient, mode)
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
        payload = data[offset:]

        if mode == CompressionMode.FLOAT32:
            expected_size = num_changed * 4
            if len(payload) != expected_size:
                raise ValueError(f"Expected {expected_size} bytes, got {len(payload)}")
            changed_values = list(struct.unpack(f'{num_changed}f', payload))

        elif mode == CompressionMode.INT16:
            expected_size = num_changed * 2
            if len(payload) != expected_size:
                raise ValueError(f"Expected {expected_size} bytes, got {len(payload)}")
            int_values = struct.unpack(f'{num_changed}h', payload)
            changed_values = [
                (val + 32768) / 65535 * self.db_range + self.min_db
                for val in int_values
            ]

        elif mode == CompressionMode.INT8:
            expected_size = num_changed
            if len(payload) != expected_size:
                raise ValueError(f"Expected {expected_size} bytes, got {len(payload)}")
            int_values = struct.unpack(f'{num_changed}b', payload)
            changed_values = [
                (val + 128) / 255 * self.db_range + self.min_db
                for val in int_values
            ]

        else:
            raise ValueError(f"Unsupported compression mode: {mode}")

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

        return band_energies, is_transient, mode

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