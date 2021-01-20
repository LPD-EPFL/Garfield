# coding: utf-8
###
 # @file   average-nan.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Synchronous average with support for NaN coordinates GAR.
###

import tensorflow as tf

from . import _GAR, register

# ---------------------------------------------------------------------------- #
# Average with support for NaN coordinates GAR class

class AverageNaNGAR(_GAR):
  """ Synchronous average with support for NaN coordinates GAR class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    pass

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    gradients = tf.parallel_stack(gradients)
    return tf.py_func(type(self)._aggregate, [gradients], gradients.dtype, stateful=False, name="GAR_average-nan")

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("average-nan", AverageNaNGAR)
