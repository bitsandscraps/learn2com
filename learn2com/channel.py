""" Channel Regularizers """
import math
import logging
from typing import List, Tuple, TypeVar
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from learn2com.debug_tools import debug_tensor

SHAPE = TypeVar('SHAPE', Tuple[int], List[int])
LOGGER = logging.getLogger(__name__)

def snr_to_stddev(snr: float) -> float:
    """ Convert `snr` to the corresponding standard deviation of a Gaussian noise """
    return 10 ** (-snr / 10) / math.sqrt(2)

def create_rotation_matrix(theta: tf.Tensor) -> tf.Tensor:
    """ Create a rotation matrix tensor """
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta),
                                tf.sin(theta), tf.cos(theta)])
    rotation_matrix = tf.reshape(rotation_matrix, [2, 2])
    return rotation_matrix

def conv1d(inputs: tf.Tensor, filter_: tf.Tensor, data_format: str = 'channels_first') -> tf.Tensor:
    """ Convolve filter_(1D) with each row of inputs

    inputs: Tensor of shape [batch_size, number_of_channels, signal_length]
    filter_: Tensor of shape [filter_length]
    """
    filter_: tf.Tensor = K.expand_dims(filter_)
    filter_: tf.Tensor = K.expand_dims(filter_)
    debug_tensor(LOGGER, filter_, 'c1d.fil')
    zero: tf.Tensor = K.constant(np.zeros(filter_.shape))
    debug_tensor(LOGGER, zero, 'c1d.zero')
    filter_first_row = K.concatenate([filter_, zero], axis=1)
    debug_tensor(LOGGER, filter_first_row, 'c1d.ffr')
    filter_second_row = K.concatenate([zero, filter_], axis=1)
    debug_tensor(LOGGER, filter_second_row, 'c1d.fsr')
    full_filter: tf.Tensor = K.concatenate([filter_first_row, filter_second_row], axis=2)
    debug_tensor(LOGGER, full_filter, 'c1d.ff')
    return K.conv1d(inputs, full_filter, data_format=data_format)

class Cast(tf.keras.layers.Layer):
    """ Casts the input layer """
    def __init__(self, dtype: tf.DType, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dtype = dtype

    def call(self, inputs: tf.Tensor, *_, **__) -> tf.Tensor:
        return K.cast(inputs, self._dtype)

    def compute_output_shape(self, input_shape: SHAPE) -> SHAPE:
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['dtype'] = self._dtype
        return config
    

class Normalize(tf.keras.layers.Layer):
    """ Normalizes the input layer """
    def call(self, inputs: tf.Tensor, *_, **__) -> tf.Tensor:
        return K.l2_normalize(inputs, axis=-1)

    def compute_output_shape(self, input_shape: SHAPE) -> SHAPE:
        return input_shape

class GaussianNoise(tf.keras.layers.Layer):
    """ Modified version of tf.keras.layers.GaussianNoise so that it works even when testing """
    def __init__(self, snr: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stddev = snr_to_stddev(snr)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        noise = K.random_normal(shape=K.shape(inputs), mean=0., stddev=self._stddev)
        return inputs + noise

    def compute_output_shape(self, input_shape: SHAPE) -> SHAPE:
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['stddev'] = self._stddev
        return config

class Delay(tf.keras.layers.Layer):
    """ Adds delay to the input layer """
    def __init__(self, taps: int, *args, **kwargs):
        self._taps = taps
        super().__init__(*args, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        debug_tensor(LOGGER, inputs, 'Del.in')
        delay: tf.Tensor = K.random_uniform([self._taps], -1.0, 1.0)
        debug_tensor(LOGGER, delay, 'Del.del')
        result: tf.Tensor = conv1d(inputs, delay)
        debug_tensor(LOGGER, result, 'Del.rslt')
        return result

    def compute_output_shape(self, input_shape: SHAPE) -> SHAPE:
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['taps'] = self._taps
        return config

class PhaseVariance(tf.keras.layers.Layer):
    """ Adds phase shift to the input layer """
    def __init__(self, bound: float, *args, **kwargs):
        self._bound = bound
        super().__init__(*args, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        delay: tf.Tensor = K.random_uniform([], 0, self._bound)
        rotation_matrix = create_rotation_matrix(delay)
        result: tf.Tensor = K.dot(rotation_matrix, inputs)
        return K.permute_dimensions(result, [1, 0, 2])

    def compute_output_shape(self, input_shape: SHAPE) -> SHAPE:
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['bound'] = self._bound
        return config
