import numpy as np


def QComplexInt8(complex_data, signed=True):
    """
    Quantizes a complex numpy array to int8 (both real and imaginary parts).

    Parameters:
        complex_data: A numpy array of complex numbers (real + imag).
        signed: Whether to use signed int8 (-128 to 127) or unsigned (0 to 255).

    Returns:
        quantized_complex: Complex array with quantized int8 values.
    """
    # Separate real and imaginary parts
    real_part = np.real(complex_data)
    imag_part = np.imag(complex_data)

    # Determine scaling factor based on the max of both real and imaginary parts
    real_min, real_max = np.min(real_part), np.max(real_part)
    imag_min, imag_max = np.min(imag_part), np.max(imag_part)

    real_range = real_max - real_min
    imag_range = imag_max - imag_min

    # Determine the quantization levels (int8 range: -128 to 127)
    if signed:
        levels = 127
        offset = -128
    else:
        levels = 255
        offset = 0

    # Scale real and imaginary parts to int8 range
    real_scaled = np.clip(np.round((real_part - real_min) / real_range * levels) + offset, -128 if signed else 0,
                          127 if signed else 255)
    imag_scaled = np.clip(np.round((imag_part - imag_min) / imag_range * levels) + offset, -128 if signed else 0,
                          127 if signed else 255)

    # Combine real and imaginary parts into a quantized complex array
    quantized_complex = real_scaled + 1j * imag_scaled
    return quantized_complex


def DQInt8Complex(quantized_complex, signed=True):
    """
    Dequantizes a complex numpy array from int8 back to float32.

    Parameters:
        quantized_complex: The quantized complex array (int8).
        signed: Whether the quantization was signed (int8 range: -128 to 127) or unsigned (0 to 255).

    Returns:
        dequantized_complex: Complex array of dequantized float32 values.
    """
    # Separate real and imaginary parts
    real_scaled = np.real(quantized_complex)
    imag_scaled = np.imag(quantized_complex)

    # Determine the scaling factor for dequantization
    if signed:
        levels = 127
        offset = -128
    else:
        levels = 255
        offset = 0

    # Calculate the real_min, real_max, imag_min, imag_max based on the quantized complex
    real_min, real_max = np.min(real_scaled), np.max(real_scaled)
    imag_min, imag_max = np.min(imag_scaled), np.max(imag_scaled)

    # Dequantize the real and imaginary parts
    real_dequantized = (real_scaled - offset) / levels * (real_max - real_min) + real_min
    imag_dequantized = (imag_scaled - offset) / levels * (imag_max - imag_min) + imag_min

    # Combine real and imaginary parts back into a complex array
    dequantized_complex = real_dequantized + 1j * imag_dequantized
    return dequantized_complex


import numpy as np


def quantize_complex(data, dtype=np.int8):
    """
    Quantizes complex float data to int8 or int16.

    Parameters:
        data (np.ndarray): Input array of complex floats.
        dtype (type): Target integer type (np.int8 or np.int16).

    Returns:
        np.ndarray: Quantized complex data.
    """
    assert np.issubdtype(dtype, np.integer), "dtype must be an integer type (e.g., np.int8 or np.int16)."

    # Determine the maximum absolute value for the chosen integer type
    max_val = np.iinfo(dtype).max  # Maximum value for the integer type
    min_val = np.iinfo(dtype).min  # Minimum value for the integer type

    # Scale real and imaginary parts independently
    real_scaled = np.clip(np.round(data.real * max_val), min_val, max_val).astype(dtype)
    imag_scaled = np.clip(np.round(data.imag * max_val), min_val, max_val).astype(dtype)

    # Combine back into complex format
    quantized_data = real_scaled + 1j * imag_scaled

    return quantized_data

