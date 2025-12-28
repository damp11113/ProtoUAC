import struct


PARAM_RANGES = {
    "IID":  (-20.0, 20.0),
    "IPD":  (-10.0, 10.0),
    "ICC":  (0.1, 1.0),
    "ICLD": (-20.0, 20.0),
}

class PSCodebook:
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
    0.5: PSCodebook({  # 0.5 BYTE (4 bits) - half byte
        "IID":  1,
        "IPD":  1,
        "ICC":  1,
        "ICLD": 1,
    }),

    1: PSCodebook({  # 1 BYTE (8 bits)
        "IID":  3,
        "IPD":  2,
        "ICC":  2,
        "ICLD": 1,
    }),

    2: PSCodebook({  # 2 BYTES (16 bits)
        "IID":  5,
        "IPD":  4,
        "ICC":  4,
        "ICLD": 3,
    }),

    4: PSCodebook({  # 4 BYTES (32 bits)
        "IID":  8,
        "IPD":  8,
        "ICC":  8,
        "ICLD": 8,
    }),

    8: PSCodebook({  # 8 BYTES (64 bits) — high precision
        "IID":  16,
        "IPD":  16,
        "ICC":  16,
        "ICLD": 16,
    }),
}

class PSScaler:
    @staticmethod
    def encode(params, nbytes):
        """
        params: dict with real values
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
    out = bytearray()

    for obj in harmonic_objects:
        out += struct.pack(">H", int(obj["freq"]))
        
        n_harmonics = len(obj["harmonics"])
        out += struct.pack(">B", n_harmonics)

        if nbytes == 0.5:
            # Pack two harmonics per byte
            nibbles = []
            for h in obj["harmonics"]:
                nibble = PSScaler.encode(h, nbytes)
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
                encoded = PSScaler.encode(h, nbytes)
                out += encoded   # length == nbytes

    return bytes(out)

def unpackObj(data, nbytes):
    idx = 0
    objects = []

    while idx < len(data):
        freq = struct.unpack_from(">H", data, idx)[0]
        idx += 2

        n = data[idx]
        idx += 1

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
                
                h = PSScaler.decode(nibble, nbytes)
                harmonics.append(h)
            
            # Move index past all packed bytes
            idx += (n + 1) // 2
        else:
            for _ in range(n):
                chunk = data[idx:idx+nbytes]
                idx += nbytes

                h = PSScaler.decode(chunk)
                harmonics.append(h)

        objects.append({
            "freq": freq,
            "n_harmonic": n,
            "harmonics": harmonics
        })

    return objects