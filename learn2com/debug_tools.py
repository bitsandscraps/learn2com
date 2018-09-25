""" Tools for debugging """

from logging import Logger
from typing import List
from numpy import ndarray
from tensorflow import Tensor

def debug_tensors(logger: Logger, tensors: List[Tensor], name: str):
    """ Print debug information about the given tensors """
    logger.debug('%s:', name)
    for tensor in tensors:
        logger.debug('            name=%-30s dtype=%s\tshape=%s', tensor.name, tensor.dtype, tensor.shape)

def debug_tensor(logger: Logger, tensor: Tensor, name: str):
    """ Print debug information about the given tensor """
    logger.debug('%-10s: name=%-30s dtype=%s\tshape=%s', name, tensor.name, tensor.dtype, tensor.shape)

def debug_array(logger: Logger, array: ndarray, name: str):
    """ Print debug information about the given layer """
    logger.debug('%-10s: name=%-30s dtype=%s\tshape=%s', name, 'N/A', array.dtype, array.shape)
