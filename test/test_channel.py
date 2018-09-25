""" Test module for learn2com.channel """

from typing import Callable
import numpy as np
import tensorflow as tf
from learn2com.channel import Normalize, conv1d

def compute_layer(layer: Callable, input_data: np.ndarray) -> np.ndarray:
    """ Feeds `layer` with `input_data` and returns the output """
    input_shape = input_data.shape[1:]
    model = tf.keras.Sequential()
    model.add(layer(input_shape=input_shape))
    return model.predict(input_data)

def test_normalize():
    """ Test Normalize layer """
    np.random.seed(0)
    input_set: np.ndarray = np.random.rand(100, 2, 128)
    output = compute_layer(Normalize, input_set)
    norm: np.ndarray = np.linalg.norm(input_set, axis=2, keepdims=True)
    true_output: np.ndarray = input_set / norm
    assert np.allclose(output, true_output)

def test_conv1d():
    """ Test conv1d """
    tf.reset_default_graph()
    filter_np: np.ndarray = np.array([1, -1])
    input_np: np.ndarray = np.arange(20).reshape((2, 2, 5))
    filter_tf: tf.Tensor = tf.constant(filter_np, dtype=tf.float32)
    input_tf: tf.Tensor = tf.constant(input_np, dtype=tf.float32)
    result_tf: tf.Tensor = conv1d(input_tf, filter_tf)
    with tf.Session() as sess:
        result_tf_run: np.ndarray = sess.run(result_tf)
    result_np: np.ndarray = np.empty((2, 2, 4))
    for idx in range(2):
        for jdx in range(2):
            result_np[idx, jdx, :] = np.convolve(input_np[idx, jdx, :], np.flip(filter_np, axis=0), mode='valid')
    print('in', input_np)
    print('np', result_np)
    print('tf', result_tf_run)
    assert np.allclose(result_np, result_tf_run)
