""" Test module for learn2com.model """

import numpy as np
import tensorflow as tf
import logging
from learn2com.main import parse_args
from learn2com.model import create_models, metric_bit_error_rate
from learn2com.debug_tools import debug_array

LOGGER = logging.getLogger(__name__)

def test_model():
    """ Test if any errors occur while manipulating models """
    tf.reset_default_graph()
    input_bits = 8
    training_set = np.ones((100, input_bits))
    training_set = training_set.astype(dtype=np.bool_)
    debug_array(LOGGER, training_set, 'train')
    kwargs = parse_args()
    kwargs['training_snr'] = kwargs['snr']
    kwargs['test_snrs'] = list(range(10))
    kwargs['input_bits'] = input_bits
    training_model, _ = create_models(**kwargs)
    with tf.Session():
        training_model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse',
                               metrics=[metric_bit_error_rate])
        training_model.fit(training_set, training_set, batch_size=1, epochs=1)
    assert True
