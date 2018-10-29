""" Module creating keras model for learn2com """

import logging
from typing import List, Tuple, Iterable
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Dropout, Input, Permute, Reshape, Flatten
from tensorflow.keras.regularizers import l2
from learn2com.channel import Normalize, GaussianNoise, Delay, PhaseVariance
from learn2com.debug_tools import debug_tensor, debug_tensors, debug_array

LOGGER = logging.getLogger(__name__)
REGULARIZE = True

def create_layers(layer: Layer, previous_layers: List[Layer]) -> List[Layer]:
    """ Apply the new layer to the list of previous layers """
    return [layer(previous_layer) for previous_layer in previous_layers]


def create_models(input_bits: int, *,
                  training_snr: float,
                  test_snrs: Iterable[float],
                  dropout_rate: float,
                  taps: int,
                  phase_bound: float,
                  l2_reg: float,
                  activation: str,
                  **_) -> Tuple[Layer, List[Layer]]:
    """ Create training and test model for learn2com

    Nontrivial arguments:
     - training_snr: snr to be used for training
     - test_snrs: list of snrs to be used for testing

    Returns: (training_model, test_model)
             number of output layers of test_model == len(test_snrs)
    """
    input_layer: Layer
    last_layer: Layer
    input_layer = last_layer = Input(shape=(input_bits,))
    debug_tensor(LOGGER, input_layer, 'input')

    # encoder
    encoder_structure: List[Tuple[str, Layer]]
    encoder_structure = [
        ('dense0', Dense(256, activation=activation, kernel_regularizer=l2(l2_reg))),
        ('dense1', Dense(256, activation=activation, kernel_regularizer=l2(l2_reg))),
        ('dense2', Dense(256, activation=activation, kernel_regularizer=l2(l2_reg))),
        ('reshape', Reshape((256, 1))),
        ('conv0', Conv1D(2, 11, activation=activation, padding='same',
                         kernel_regularizer=l2(l2_reg))),
        ('permute', Permute((2, 1))),
    ]

    for name, layer in encoder_structure:
        last_layer: Layer = layer(last_layer)
        debug_tensor(LOGGER, last_layer, name)

    if REGULARIZE:
        # add regularizers
        last_layer = Normalize()(last_layer)
        debug_tensor(LOGGER, last_layer, 'normalize')

        training_last_layer = GaussianNoise(training_snr)(last_layer)
        debug_tensor(LOGGER, last_layer, 'awgn')

        last_layers = [GaussianNoise(snr)(last_layer) for snr in test_snrs]
        # add training_last_layer to last_layers just for convenience
        last_layers.insert(0, training_last_layer)

        regularizer_structure: List[Tuple[str, Layer]]
        regularizer_structure = [
            ('dropout', Dropout(dropout_rate)),
            # ('delay', Delay(taps)),
            ('pv', PhaseVariance(phase_bound)),
        ]

        for name, layer in regularizer_structure:
            last_layers = create_layers(layer, last_layers)
            debug_tensors(LOGGER, last_layers, name)

    else:
        last_layers = [last_layer, last_layer, last_layer]

    # decoder
    decoder_architecture = [
        ('conv1', Conv1D(4, 11, data_format='channels_first', padding='same',
                         activation=activation, kernel_regularizer=l2(l2_reg))),
        ('flatten', Flatten(data_format='channels_first')),
        ('dense3', Dense(256, activation=activation, kernel_regularizer=l2(l2_reg))),
        ('dense4', Dense(256, activation=activation, kernel_regularizer=l2(l2_reg))),
        ('dense5', Dense(input_bits, activation=None, kernel_regularizer=l2(l2_reg))),
    ]

    for name, layer in decoder_architecture:
        last_layers = create_layers(layer, last_layers)
        debug_tensors(LOGGER, last_layers, name)

    training_last_layer = last_layers[0]
    test_last_layers = last_layers[1:]

    # create models
    training_model = Model(inputs=input_layer, outputs=training_last_layer)
    test_model = Model(inputs=input_layer, outputs=test_last_layers)

    return training_model, test_model

def metric_bit_error_rate(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Custom  metric function """
    debug_tensor(LOGGER, y_true, 'ber_true')
    debug_tensor(LOGGER, y_pred, 'ber_pred')
    threshold_np: np.ndarray = np.ones(y_pred.shape[-1:]) * 0.5
    debug_array(LOGGER, threshold_np, 'ber_np')
    threshold: tf.Tensor = K.constant(threshold_np.tolist())
    debug_tensor(LOGGER, threshold, 'ber_thr')
    decoded: tf.Tensor = K.greater(y_pred, threshold)
    debug_tensor(LOGGER, decoded, 'ber_dec')
    ytrue_cast: tf.Tensor = K.greater(y_true, threshold)
    debug_tensor(LOGGER, ytrue_cast, 'ber_tr_c')
    compare: tf.Tensor = K.equal(decoded, ytrue_cast)
    debug_tensor(LOGGER, compare, 'ber_comp')
    result = 1 - K.mean(compare)
    debug_tensor(LOGGER, result, 'ber_rslt')
    return result
