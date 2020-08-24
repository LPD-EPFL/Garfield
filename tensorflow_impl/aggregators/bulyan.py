# coding: utf-8
###
 # @file   bulyan.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Bulyan over Multi-Krum GAR.
###

import tensorflow as tf
import warnings

import tools
import native
from . import _GAR, register, deprecated_native

# ---------------------------------------------------------------------------- #
# Bulyan GAR class

class PYBulyanGAR(_GAR):
  """ Full-Python/(deprecated) native Bulyan of Multi-Krum GAR class.
  """

  def _aggregate(self, gradients):
    """ Aggregate the gradient using the associated (deprecated) native helper.
    Args:
      gradients Stacked list of submitted gradients, as a numpy array
    Returns:
      Aggregated gradient, as a numpy array
    """
    return deprecated_native.bulyan(gradients, self.__f, self.__s)

  def __init__(self, nbworkers, nbbyzwrks, args):
    warnings.warn("Python/native implementation of Bulyan has been deprecated in favor of the CO implementations", category=DeprecationWarning, stacklevel=3)
    self.__f = nbbyzwrks
    self.__s = nbworkers - 2 * nbbyzwrks - 2

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    gradients = tf.parallel_stack(gradients)
    return tf.py_func(self._aggregate, [gradients], gradients.dtype, stateful=False, name="GAR_bulyan")

class COBulyanGAR(_GAR):
  """ Full-custom operation Bulyan of Multi-Krum GAR class.
  """

  # Name of the associated custom operation
  co_name = "bulyan"

  def __init__(self, nbworkers, nbbyzwrks, args):
    self.__nbworkers = nbworkers
    self.__nbbyzwrks = nbbyzwrks
    self.__multikrum = nbworkers - nbbyzwrks - 2

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return native.instantiate_op(type(self).co_name, tf.parallel_stack(gradients), f=self.__nbbyzwrks, m=self.__multikrum)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rules
register("bulyan-py", PYBulyanGAR)
if COBulyanGAR.co_name in native.itemize_op():
  register("bulyan", COBulyanGAR)
else:
  tools.warning("GAR 'bulyan' could not be registered since the associated custom operation " + repr(COBulyanGAR.co_name) + " is unavailable")
