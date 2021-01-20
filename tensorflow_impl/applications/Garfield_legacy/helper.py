# coding: utf-8
###
 # @file   helper.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
###

import tensorflow as tf

# ---------------------------------------------------------------------------- #
# PS-worker device setter producer

# Operation types that goes on the parameter server
_ps_ops = ("Variable", "VariableV2", "VarHandleOp", "AutoReloadVariable",
           "MutableHashTableV2", "MutableDenseHashTableV2", "MutableHashTable",
           "MutableHashTableOfTensorsV2", "MutableDenseHashTable",
           "BoostedTreesEnsembleResourceHandleOp", "MutableHashTableOfTensors")

def replica_device_setter(device_ps, device_wk):
  """ Generate a PS-worker device setter.
  Args:
    device_ps Parameter server device name/function
    device_wk Current worker device name/function
  Returns:
    Device setter closure
  """
  def setter(op):
    global _ps_ops
    if op.type in _ps_ops:
      if callable(device_ps):
        return device_ps(op)
      return device_ps
    else:
      if callable(device_wk):
        return device_wk(op)
      return device_wk
  return setter

# ---------------------------------------------------------------------------- #
# l1/l2 regularization helpers

def regularization(norm):
  """ Compute the regularization loss.
  Args:
    norm Norm to use (i.e. 1 or 2)
  Returns:
    Regularization loss
  """
  # Loss computation
  variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  if norm == 1:
    return tf.reduce_sum([tf.reduce_sum(tf.abs(variable)) for variable in variables], name="l1_loss")
  elif norm == 2:
    return tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(variable)) for variable in variables]), name="l2_loss")
  else:
    "Invalid value " + repr(norm) + " for parameter 'norm'"

# ---------------------------------------------------------------------------- #
# Gradient flattening-inflating helpers

def flatten(tensors, flatmap=None):
  """ Flatten the tensor from the list of (tensor, variable).
  Args:
    tensors List of (tensor, variable)
    flatmap Mapping between variables and their gradient's position in the flattened tensor (optional, build it if None)
  Returns:
    Flattened tensor, mapping variable/position (only if 'flatmap' was None)
  """
  with tf.name_scope("flatten"):
    if flatmap is None:
      flatmap = {}
      res = []
      for gradient, variable in tensors:
        if gradient is None:
          continue
        flatmap[variable] = len(res)
        res.append(tf.reshape(gradient, (-1,)))
      return tf.concat(res, 0), flatmap
    else:
      res = [None] * len(flatmap)
      for gradient, variable in tensors:
        if gradient is None:
          continue
        res[flatmap[variable]] = tf.reshape(gradient, (-1,))
      return tf.concat(res, 0)

def mapflat(flatmap):
  """ Transform a map variable -> gradient position into the associated ordered list of variables.
  Args:
    flatmap Mapping between variables and their gradient's position in the flattened tensor
  Returns:
    List of variables in the order defined by their respective position in the flat gradient
  """
  res = [None] * len(flatmap)
  for variable, position in flatmap.items():
    res[position] = variable
  return res

def inflate(tensor, mapflat):
  """ Inflate the tensor to a list of (tensor, variable).
  Args:
    tensor  Flattened tensor
    mapflat List of variables in the order defined by their respective position in the flat gradient
  Returns:
    List of (tensor, variable)
  """
  res = []
  pos = 0
  with tf.name_scope("inflate"):
    for variable in mapflat:
      shape = variable.shape
      size = shape.num_elements()
      tnsr = tf.reshape(tf.slice(tensor, [pos], [size]), shape)
      res.append((tnsr, variable))
      pos += size
  return res

# ---------------------------------------------------------------------------- #
