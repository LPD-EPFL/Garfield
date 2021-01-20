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
 # Base gradient aggregation rule class (GAR) and loading of the local modules.
###

import pathlib

import tools

# ---------------------------------------------------------------------------- #
# Base gradient aggregation rule class

class _GAR:
  """ Base gradient aggregation rule class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    """ Unimplemented constructor, no graph available at this time.
    Args:
      nbworkers Total number of workers
      nbbyzwrks Declared number of Byzantine workers
      args      Command line argument list
    """
    raise NotImplementedError

  def aggregate(self, gradients):
    """ Build the gradient aggregation operation of the given gradients.
    Args:
      gradients Computed gradient tensors
    Returns:
      Aggregated gradient tensor
    """
    raise NotImplementedError

# ---------------------------------------------------------------------------- #
# GAR script loader

# Register instance
_register   = tools.ClassRegister("GAR")
itemize     = _register.itemize
register    = _register.register
instantiate = _register.instantiate
del _register

# Load all local modules
with tools.Context("aggregators", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())
