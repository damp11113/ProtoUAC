
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

    8: PSCodebook({  # 8 BYTES (64 bits) – high precision
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
        nbytes: 1, 2, 4, or 8
        """
        cb = CODEBOOKS[nbytes]
        q = cb.quantize(params)

        payload = 0
        bitpos = 0
        for k, nbits in cb.bits.items():
            payload |= (q[k] << bitpos)
            bitpos += nbits

        return payload.to_bytes(cb.total_bytes, "big")

    @staticmethod
    def decode(data):
        """
        data: raw bytes (length determines codebook)
        """
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
        out += struct.pack(">B", len(obj["harmonics"]))

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