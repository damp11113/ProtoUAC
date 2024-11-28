t_index = [
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8]  # index table

t_step = [
    7, 8, 9, 10, 11, 12, 13, 14,
    16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658,
    724, 796, 876, 963, 1060, 1166, 1282, 1411,
    1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484,
    7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767]  # quantize table

_encoder_predicted = 0
_encoder_index = 0
_encoder_step = 7
_decoder_predicted = 0
_decoder_index = 0
_decoder_step = 7


def encode_sample(sample):
    # encode one linear pcm sample to ima adpcm neeble
    # using global encoder state
    global _encoder_predicted
    global _encoder_index

    delta = sample - _encoder_predicted

    if delta >= 0:
        value = 0
    else:
        value = 8
        delta = -delta

    step = t_step[_encoder_index]

    diff = step >> 3

    if delta > step:
        value |= 4
        delta -= step
        diff += step
    step >>= 1

    if delta > step:
        value |= 2
        delta -= step
        diff += step
    step >>= 1

    if delta > step:
        value |= 1
        diff += step

    if value & 8:
        _encoder_predicted -= diff
    else:
        _encoder_predicted += diff

    if _encoder_predicted < - 0x8000:
        _encoder_predicted = -0x8000
    elif _encoder_predicted > 0x7fff:
        _encoder_predicted = 0x7fff

    _encoder_index += t_index[value & 7]

    if _encoder_index < 0:
        _encoder_index = 0
    elif _encoder_index > 88:
        _encoder_index = 88

    return value


def decode_sample(neeble):
    # decode one sample from compressed neeble
    # using global decoder state
    global _decoder_predicted
    global _decoder_index
    global _decoder_step

    difference = 0

    if neeble & 4:
        difference += _decoder_step

    if neeble & 2:
        difference += _decoder_step >> 1

    if neeble & 1:
        difference += _decoder_step >> 2

    difference += _decoder_step >> 3

    if neeble & 8:
        difference = -difference

    _decoder_predicted += difference

    if _decoder_predicted > 32767:
        _decoder_predicted = 32767

    elif _decoder_predicted < -32767:
        _decoder_predicted = - 32767

    _decoder_index += t_index[neeble]
    if _decoder_index < 0:
        _decoder_index = 0
    elif _decoder_index > 88:
        _decoder_index = 88
    _decoder_step = t_step[_decoder_index]

    return _decoder_predicted
