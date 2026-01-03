import struct
import numpy as np


PARAM_RANGES = {
    "amp":   (0.0, 1.0),      # Amplitude range
    "phase": (-np.pi, np.pi), # Phase range
}

class PHXCCodebook:
    def __init__(self, bits):
        """
        bits: dict { param_name: bit_count }
        """
        self.bits = bits
        self.total_bits = sum(bits.values())
        self.total_bytes = (self.total_bits + 7) // 8

    def quantize(self, params):
        q = {}
        for k, nbits in self.bits.items():
            smin, smax = PARAM_RANGES[k]
            levels = (1 << nbits) - 1

            v = max(min(params[k], smax), smin)
            norm = (v - smin) / (smax - smin)
            q[k] = int(round(norm * levels))
        return q

    def dequantize(self, q):
        params = {}
        for k, nbits in self.bits.items():
            smin, smax = PARAM_RANGES[k]
            levels = (1 << nbits) - 1

            norm = q[k] / levels if levels > 0 else 0.0
            params[k] = smin + norm * (smax - smin)
        return params

CODEBOOKS = {
    0.5: PHXCCodebook({  # 0.5 BYTE (4 bits) - half byte
        "amp":   2,
        "phase": 2,
    }),

    1: PHXCCodebook({  # 1 BYTE (8 bits)
        "amp":   4,
        "phase": 4,
    }),

    2: PHXCCodebook({  # 2 BYTES (16 bits)
        "amp":   8,
        "phase": 8,
    }),

    4: PHXCCodebook({  # 4 BYTES (32 bits)
        "amp":   16,
        "phase": 16,
    }),

    8: PHXCCodebook({  # 8 BYTES (64 bits) — high precision
        "amp":   32,
        "phase": 32,
    }),
}

class PHXCScaler:
    @staticmethod
    def encode(params, nbytes):
        """
        params: dict with 'amp' and 'phase' values
        nbytes: 0.5, 1, 2, 4, or 8
        """
        cb = CODEBOOKS[nbytes]
        q = cb.quantize(params)

        payload = 0
        bitpos = 0
        for k, nbits in cb.bits.items():
            payload |= (q[k] << bitpos)
            bitpos += nbits

        if nbytes == 0.5:
            # Return 4-bit value (not converted to bytes yet)
            return payload
        else:
            return payload.to_bytes(cb.total_bytes, "big")

    @staticmethod
    def decode(data, nbytes=None):
        """
        data: raw bytes (length determines codebook) or int (for 4-bit nibble)
        nbytes: optional, used when data is an int
        """
        if isinstance(data, int):
            # 4-bit nibble
            nbytes = 0.5
            cb = CODEBOOKS[nbytes]
            payload = data
        else:
            nbytes = len(data)
            cb = CODEBOOKS[nbytes]
            payload = int.from_bytes(data, "big")

        q = {}
        bitpos = 0
        for k, nbits in cb.bits.items():
            mask = (1 << nbits) - 1
            q[k] = (payload >> bitpos) & mask
            bitpos += nbits

        return cb.dequantize(q)

def packObj(harmonic_objects, nbytes):
    """
    Pack PHXC harmonic objects into binary format.
    
    Format per object:
    - freq: 2 bytes (unsigned short, Hz)
    - n_harmonics: 1 byte
    - main_amp: 4 bytes (float32)
    - harmonics: n_harmonics × nbytes (each containing amp + phase)
    """
    out = bytearray()

    for obj in harmonic_objects:
        # Pack frequency (uint16)
        out += struct.pack(">H", int(obj["freq"]))
        
        # Pack number of harmonics (uint8)
        n_harmonics = len(obj["harmonics"])
        out += struct.pack(">B", n_harmonics)
        
        # Pack main amplitude (float32)
        out += struct.pack(">f", float(obj.get("main_amp", 1.0)))

        if nbytes == 0.5:
            # Pack two harmonics per byte
            nibbles = []
            for h in obj["harmonics"]:
                nibble = PHXCScaler.encode(h, nbytes)
                nibbles.append(nibble)
            
            # Pack nibbles into bytes (2 nibbles per byte)
            for i in range(0, len(nibbles), 2):
                if i + 1 < len(nibbles):
                    # Two nibbles: high nibble | low nibble
                    byte_val = (nibbles[i] << 4) | nibbles[i + 1]
                else:
                    # Odd number of harmonics: pad with 0
                    byte_val = (nibbles[i] << 4)
                out.append(byte_val)
        else:
            for h in obj["harmonics"]:
                encoded = PHXCScaler.encode(h, nbytes)
                out += encoded   # length == nbytes

    return bytes(out)

def unpackObj(data, nbytes):
    """
    Unpack PHXC binary data back into harmonic objects.
    """
    idx = 0
    objects = []

    while idx < len(data):
        # Unpack frequency
        freq = struct.unpack_from(">H", data, idx)[0]
        idx += 2

        # Unpack number of harmonics
        n = data[idx]
        idx += 1
        
        # Unpack main amplitude
        main_amp = struct.unpack_from(">f", data, idx)[0]
        idx += 4

        harmonics = []
        
        if nbytes == 0.5:
            # Unpack nibbles (2 per byte)
            for i in range(n):
                byte_idx = idx + (i // 2)
                byte_val = data[byte_idx]
                
                if i % 2 == 0:
                    # High nibble (first harmonic in byte)
                    nibble = (byte_val >> 4) & 0x0F
                else:
                    # Low nibble (second harmonic in byte)
                    nibble = byte_val & 0x0F
                
                h = PHXCScaler.decode(nibble, nbytes)
                harmonics.append(h)
            
            # Move index past all packed bytes
            idx += (n + 1) // 2
        else:
            for _ in range(n):
                chunk = data[idx:idx+nbytes]
                idx += nbytes

                h = PHXCScaler.decode(chunk)
                harmonics.append(h)

        objects.append({
            "freq": float(freq),
            "n_harmonic": n,
            "main_amp": float(main_amp),
            "harmonics": harmonics
        })

    return objects

