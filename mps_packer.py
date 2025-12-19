import math
import struct

PS_BAND_RANGES = {
    "PAN": (-1.0, 1.0),
    "IPD": (-math.pi, math.pi),
}

class PSBandCodebook:
    def __init__(self, pan_bits, ipd_bits):
        self.pan_bits = pan_bits
        self.ipd_bits = ipd_bits
        self.ic_bits  = 1

        self.total_bits = pan_bits + ipd_bits + 1
        self.total_bytes = (self.total_bits + 7) // 8

    def quantize(self, pan, ipd, ic):
        # PAN
        pmin, pmax = PS_BAND_RANGES["PAN"]
        plevels = (1 << self.pan_bits) - 1
        pan = max(min(pan, pmax), pmin)
        pan_q = int(round((pan - pmin) / (pmax - pmin) * plevels))

        # IPD
        imin, imax = PS_BAND_RANGES["IPD"]
        ilevels = (1 << self.ipd_bits) - 1
        ipd = max(min(ipd, imax), imin)
        ipd_q = int(round((ipd - imin) / (imax - imin) * ilevels))

        # IC (boolean)
        ic_q = 1 if ic else 0

        return pan_q, ipd_q, ic_q

    def dequantize(self, pan_q, ipd_q, ic_q):
        pmin, pmax = PS_BAND_RANGES["PAN"]
        plevels = (1 << self.pan_bits) - 1
        pan = pmin + (pan_q / plevels) * (pmax - pmin)

        imin, imax = PS_BAND_RANGES["IPD"]
        ilevels = (1 << self.ipd_bits) - 1
        ipd = imin + (ipd_q / ilevels) * (imax - imin)

        ic = bool(ic_q)

        return pan, ipd, ic

BAND_CODEBOOKS = {
    1: PSBandCodebook(pan_bits=4,  ipd_bits=3),   # 4+3+1 = 8 bits
    2: PSBandCodebook(pan_bits=6,  ipd_bits=9),   # 6+9+1 = 16 bits
    4: PSBandCodebook(pan_bits=12, ipd_bits=19),  # 12+19+1 = 32 bits
}

class PSBandScaler:
    @staticmethod
    def encode_band(pan, ipd, ic, nbytes):
        cb = BAND_CODEBOOKS[nbytes]
        pan_q, ipd_q, ic_q = cb.quantize(pan, ipd, ic)

        payload = (
            (pan_q << (cb.ipd_bits + 1)) |
            (ipd_q << 1) |
            ic_q
        )

        return payload.to_bytes(cb.total_bytes, "big")

    @staticmethod
    def decode_band(data):
        cb = BAND_CODEBOOKS[len(data)]
        payload = int.from_bytes(data, "big")

        ic_q  = payload & 0x1
        ipd_q = (payload >> 1) & ((1 << cb.ipd_bits) - 1)
        pan_q = payload >> (cb.ipd_bits + 1)

        return cb.dequantize(pan_q, ipd_q, ic_q)

def packObj(results, min_freq, max_freq, point, nbytes):
    out = bytearray()

    n_bands = len(results)
    if not (0 <= point <= 255):
        raise ValueError("point must fit in uint8")
    if not (0 <= n_bands <= 255):
        raise ValueError("n_bands must fit in uint8")

    # Header
    out += struct.pack("<i", int(min_freq))
    out += struct.pack("<i", int(max_freq))
    out.append(point & 0xFF)
    out.append(n_bands & 0xFF)
    out.append(nbytes & 0xFF)

    # Bands
    for _, pan, ipd, ic in results:
        out += PSBandScaler.encode_band(pan, ipd, ic, nbytes)

    return bytes(out)

def unpackObj(data):
    idx = 0

    min_freq = struct.unpack_from("<i", data, idx)[0]
    idx += 4
    max_freq = struct.unpack_from("<i", data, idx)[0]
    idx += 4

    point   = data[idx]; idx += 1
    n_bands = data[idx]; idx += 1
    nbytes  = data[idx]; idx += 1

    output = []

    for _ in range(n_bands):
        chunk = data[idx:idx+nbytes]
        idx += nbytes

        pan, ipd, ic = PSBandScaler.decode_band(chunk)
        output.append((0, pan, ipd, ic))


    return output, min_freq, max_freq, point
