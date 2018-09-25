""" Learning to Communicate Main Module """

import argparse
import logging
import os.path
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from learn2com.model import create_models, metric_bit_error_rate

def bit_error_rate(ground_truth: np.ndarray, observed: np.ndarray) -> float:
    """ Calculate the bit error rate """
    logging.debug('observed.shape=%s ground_truth.shape=%s',
                  observed.shape, ground_truth.shape)
    observed = observed > 0.5
    return np.mean(np.isclose(observed, ground_truth))

def parse_args() -> dict:
    """ Parse arguments

    seed: for 100% reproducability. fed to `np.random.seed` and `tf.set_random_seed` (default=0)
    dataset_size: size of the whole data set (default=1e6)
    training_set_ratio: training_set_size / dataset_size (default=0.8)
    log_level: fed to logging.basicConfig (default=info)
    input_bits: number of bits in the input signal (default=128)
    snr: signal to noise ratio in dB (default=5.0)
    dropout_rate: probability a node is dropped out (default=0.01)
    taps: the length of a delay filter (default = 1)
    phase_bound: bound of phase variance (default = 0.01)
    savedir: directory to save the model (default=model)
    logdir: directory to save logs (default=log)
    epochs: number of epochs to train (default=50)
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset-size', type=int, default=1000000)
    parser.add_argument('--training-set-ratio', type=float, default=0.8)
    parser.add_argument('--log-level', type=str, default='info')
    parser.add_argument('--input-bits', type=int, default=128)
    parser.add_argument('--snr', type=float, default=5.0)
    parser.add_argument('--dropout-rate', type=float, default=0.01)
    parser.add_argument('--taps', type=int, default=1)
    parser.add_argument('--phase-bound', type=float, default=0.01)
    parser.add_argument('--savedir', type=str, default='model')
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--l2-reg', type=float, default=0.01)
    # change log_level to the corresponding numeric level
    args, _ = parser.parse_known_args()
    dict_args = vars(args)
    log_level: str = dict_args['log_level']
    dict_args['log_level'] = getattr(logging, log_level.upper())
    return dict_args

def main(seed: int, dataset_size: int, training_set_ratio: float,
         log_level: int, input_bits: int, snr: float, dropout_rate: float,
         taps: int, phase_bound: float, savedir: str, logdir: str, epochs: int, l2_reg: float) -> None:
    """ The main routine: create training/test sets, create and learn a model, test it ... """
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.reset_default_graph()
    filename = 'snr-{0:.1f}-dropout-{1:.2f}-taps-{2}-phase-bound-{3:.1f}'.format(snr, dropout_rate, taps, phase_bound)
    filename = filename.replace('.', '-')
    savefile = os.path.join(savedir, filename + '.h5')
    logfile = os.path.join(logdir, filename)
    print('Start logging on ' + logfile)
    logging.basicConfig(level=log_level,
                        format='[%(asctime)-15s] %(levelname)s : %(name)s : %(message)s',
                        filename=logfile)

    logging.info('************** Settings **************')
    logging.info('seed: %d', seed)
    logging.info('dataset_size: %d', dataset_size)
    logging.info('training_set_ratio: %.02f', training_set_ratio)
    logging.info('l2_reg: %f', l2_reg)
    logging.info('input_bits: %d', input_bits)
    logging.info('snr: %.01f', snr)
    logging.info('dropout_rate: %.02f', dropout_rate)
    logging.info('taps: %d', taps)
    logging.info('phase_bound: %f', phase_bound)
    logging.info('epochs: %d', epochs)
    logging.info('savefile: %s', savefile)
    logging.info('logfile: %s', logfile)
    logging.info('*********** End of Settings ***********')

    # create training/test sets
    if training_set_ratio > 1 or training_set_ratio < 0:
        raise ValueError('training_set_ratio must be between 0 and 1')
    training_set_size = round(dataset_size * training_set_ratio)
    dataset: np.ndarray
    dataset = np.random.randint(2, size=(dataset_size, input_bits))
    dataset = dataset.astype(dtype=np.float_)
    training_set = dataset[:training_set_size, ...]
    test_set = dataset[training_set_size:, ...]
    logging.debug('training_set.shape=%s test_set.shape=%s', training_set.shape, test_set.shape)

    test_snrs = np.linspace(-10., 20., 10, endpoint=False)

    training_model, test_model = create_models(
        input_bits=input_bits, training_snr=snr, test_snrs=test_snrs,
        dropout_rate=dropout_rate, taps=taps, phase_bound=phase_bound, l2_reg=l2_reg)

    # check if savefile exists. If it exsits, there is no need to train again
    if os.path.isfile(savefile):
        training_model.load_weights(savefile)
        logging.info('Model loaded.')
    else:
        training_model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse',
                               metrics=[metric_bit_error_rate])
        training_model.fit(training_set, training_set, batch_size=64, epochs=epochs)
        training_model.save_weights(savefile)
        logging.info('Training complete.')

    # testing
    test_results = test_model.predict(test_set)
    for decoded in test_results:
        print(bit_error_rate(test_set, decoded))
    logging.info('Test complete.')

if __name__ == '__main__':
    main(**parse_args())
