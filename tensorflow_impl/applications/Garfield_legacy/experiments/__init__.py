# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Base experiment (= model + dataset) class and loading of the local modules.
###

import pathlib

import tools

# ---------------------------------------------------------------------------- #
# Base experiment class

class _Experiment:
  """ Base experiment class.
  """

  def __init__(self, args):
    """ Unimplemented constructor, no graph available at this time.
    Args:
      args Command line argument list
    """
    raise NotImplementedError

  def loss(self, device_dataset, device_models, trace=False):
    """ Build loss tensors on the specified devices, placement on parameter server by default.
    Args:
      device_dataset Dataset device name/function (same instance between calls if same task, i.e. can use 'is')
      device_models  Model device names/functions, one per worker on the associated task
      trace          Whether to add trace prints for every important step of the computations
    Returns:
      List of loss tensors associated with 'device_models'
    """
    raise NotImplementedError

  def accuracy(self, device_dataset, device_model, trace=False):
    """ Build an accuracy tensor on the specified devices, placement on parameter server by default.
    Args:
      device_dataset Dataset device name/function (same instance between calls if same task, i.e. can use 'is')
      device_models  Model device names/functions, one per worker on the associated task
      trace          Whether to add trace prints for every important step of the computations
    Returns:
      Map of metric string name -> aggregated metric tensor associated with 'device_models'
    """
    raise NotImplementedError

# ---------------------------------------------------------------------------- #
# Experiment register and loader

# Register instance
_register   = tools.ClassRegister("experiment")
itemize     = _register.itemize
register    = _register.register
instantiate = _register.instantiate
del _register

# Load all local modules
with tools.Context("experiments", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())
