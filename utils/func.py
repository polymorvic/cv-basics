import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from .image_array import NumpyImage


def _add_padding(input_array: np.ndarray, padding_width: int, padding_height: int) -> NumpyImage:
    return NumpyImage(np.pad(input_array, pad_width = ((padding_height, padding_height), (padding_width, padding_width)), mode='constant', constant_values=0))


def morph_transform(input_array: np.ndarray, kernel_array: np.ndarray, op: Literal['erode', 'dilate'] = 'erode', apply_padding: bool = True) -> np.ndarray:

    if not op.lower() in ('erode', 'dilate',):
        raise Exception(f'Only erode and dilate are supported, not {op}')

    if not ((k := NumpyImage(kernel_array)).width % 2 or k.height % 2):
        raise Exception('One of dimension should be even')

    input_array = NumpyImage(input_array)
    if apply_padding:
        kernel_array = NumpyImage(kernel_array)
        pad_w = (kernel_array.width - 1) // 2
        pad_h = (kernel_array.height - 1) // 2
        input_array = _add_padding(input_array, pad_w, pad_h)
    else:
        pad_w = 0
        pad_h = 0

    output_array = np.zeros_like(input_array)
    option_caller = {
        'erode': np.all,
        'dilate': np.any
    }
    for row_idx in range(pad_h, input_array.height - pad_h):
        for col_idx in range(pad_w, input_array.width - pad_w):
            roi = input_array[row_idx - pad_h:row_idx + pad_h + 1, col_idx - pad_w:col_idx + pad_w + 1]
            overlap = (roi > 0)[kernel_array == 1]
            result = option_caller[op](overlap)
            output_array[row_idx, col_idx] = 1 if result else 0

    return output_array[pad_h:-pad_h, pad_w:-pad_w]