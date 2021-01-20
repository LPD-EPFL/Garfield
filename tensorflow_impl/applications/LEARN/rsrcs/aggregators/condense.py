# coding: utf-8
###
 # @file   condense.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Condense parameter vector aggregation random function.
###

import tensorflow as tf

import tools
from . import _GAR, register

# ---------------------------------------------------------------------------- #
# Condense random function

class TFCondenseGAR(_GAR):
  """ Full-TensorFlow condense random function class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    # Parse key:val arguments
    ps = tools.parse_keyval([] if args is None else args, defaults={"ps": 0.9})["ps"]
    if ps <= 0 or ps > 1:
      raise tools.UserException("Invalid selection probability, got %s" % (ps,))
    # Finalization
    self._p = ps
    self._f = nbbyzwrks

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) >= 2 * self._f + 2, "Not enough gradients to aggregate, expected at least %d, got %d" % (2 * self._f + 2, len(gradients))
    # Sample selection indications
    c = tf.cast(tf.distributions.Bernoulli(probs=tf.ones_like(gradients[0]) * self._p).sample(), dtype=tf.float32)
    # Compute median
    g = tf.parallel_stack(gradients)
    m = tf.transpose(tf.reduce_min(tf.nn.top_k(tf.transpose(g), (g.shape[0] + 1) // 2, sorted=False).values, axis=1))
    # Add masked first gradient and return
    return m * c + gradients[0] * (1 - c)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("condense", TFCondenseGAR)
