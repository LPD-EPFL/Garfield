# coding: utf-8
###
 # @file   median.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Coordinate-wise median GAR.
###

import tensorflow as tf

import tools
import native
from . import _GAR, register

# ---------------------------------------------------------------------------- #
# Nan-resilient median coordinate-per-coordinate GAR class

class COMedianGAR(_GAR):
  """ Full-custom operation median GAR class.
  """

  # Name of the associated custom operation
  co_name = "median"

  def __init__(self, nbworkers, nbbyzwrks, args):
    pass

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return native.instantiate_op(type(self).co_name, tf.parallel_stack(gradients))

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
if COMedianGAR.co_name in native.itemize_op():
  register("median", COMedianGAR)
else:
  tools.warning("GAR 'median' could not be registered since the associated custom operation " + repr(COMedianGAR.co_name) + " is unavailable")
