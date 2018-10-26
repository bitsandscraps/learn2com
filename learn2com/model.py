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
REGULARIZE = False

def create_models(input_bits: int, *,
                  training_snr: float,
                  test_snrs: Iterable[float],
                  dropout_rate: float,
                  taps: int,
                  phase_bound: float,
                  l2_reg: float,
                  **_) -> Tuple[Layer, List[Layer]]:
    """ Create training and test model for learn2com

    Nontrivial arguments:
     - training_snr: snr to be used for training
     - test_snrs: list of snrs to be used for testing

    Returns: (training_model, test_model)
             number of output layers of test_model == len(test_snrs)
    """
    input_layer: Layer
    input_layer = Input(shape=(input_bits,))
    debug_tensor(LOGGER, input_layer, 'input')

    # encoder
    last_layer: Layer
    last_layer = Dense(256, activation='sigmoid')(input_layer)
    debug_tensor(LOGGER, last_layer, 'dense0')
    last_layer = Dense(256, activation='sigmoid')(last_layer)
    debug_tensor(LOGGER, last_layer, 'dense1')
    last_layer = Dense(256, activation='sigmoid')(last_layer)
    debug_tensor(LOGGER, last_layer, 'dense2')
    last_layer = Reshape((256, 1))(last_layer)
    debug_tensor(LOGGER, last_layer, 'reshape0')
    last_layer = Conv1D(2, 11, activation='sigmoid', padding='same')(last_layer)
    debug_tensor(LOGGER, last_layer, 'conv0')
    last_layer = Permute((2, 1))(last_layer)
    debug_tensor(LOGGER, last_layer, 'permute')

    if REGULARIZE:
        # add regularizers
        last_layer = Normalize()(last_layer)
        debug_tensor(LOGGER, last_layer, 'normalize')
        training_last_layer = GaussianNoise(training_snr)(last_layer)
        debug_tensor(LOGGER, last_layer, 'awgn')
        """
        training_last_layer = Dropout(dropout_rate)(training_last_layer)
        debug_tensor(LOGGER, last_layer, 'dropout')
        training_last_layer = Delay(taps)(training_last_layer)
        debug_tensor(LOGGER, last_layer, 'delay')
        training_last_layer = PhaseVariance(phase_bound)(training_last_layer)
        debug_tensor(LOGGER, last_layer, 'pv')
        """
        # test model does not need additional regularizers
        last_layers = [GaussianNoise(snr)(last_layer) for snr in test_snrs]
        # add training_last_layer to last_layers just for convenience
        last_layers.insert(0, training_last_layer)
    else:
        last_layers = [last_layer, last_layer]

    # decoder
    #last_layers = [Conv1D(4, 11, data_format='channels_first', padding='same', activation='sigmoid', kernel_regularizer=l2(l2_reg))(last_layer) for last_layer in last_layers]
    #debug_tensors(LOGGER, last_layers, 'conv1')
    last_layers = [Flatten(data_format='channels_first')(last_layer) for last_layer in last_layers]
    debug_tensors(LOGGER, last_layers, 'flatten')
    last_layers = [Dense(256, activation='sigmoid')(last_layer) for last_layer in last_layers]
    debug_tensors(LOGGER, last_layers, 'dense3')
    last_layers = [Dense(256, activation='sigmoid')(last_layer) for last_layer in last_layers]
    debug_tensors(LOGGER, last_layers, 'dense4')
    last_layers = [Dense(input_bits, activation='sigmoid')(last_layer) for last_layer in last_layers]
    debug_tensors(LOGGER, last_layers, 'dense5')
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
    result = K.mean(compare)
    debug_tensor(LOGGER, result, 'ber_rslt')
    return result
