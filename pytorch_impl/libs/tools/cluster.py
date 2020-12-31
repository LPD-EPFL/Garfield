# coding: utf-8
###
 # @file   cluster.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Basic TF cluster specification parsers and helpers.
###

__all__ = ["cluster_parsers", "cluster_parse"]

import json
import os
import pathlib

import tools

# ---------------------------------------------------------------------------- #
# G5k cluster parser

_g5k_env_key = "OAR_FILE_NODES"
_g5k_cluster = None

def _g5k_parser():
  """ Generate the cluster specification from the G5k-specific cluster specification file.
  Returns:
    Cluster dictionary, with only 1 ps and n-1 worker(s), all using port 7000
  """
  global _g5k_env_key
  global _g5k_cluster
  if _g5k_cluster is not None:
    return _g5k_cluster
  if _g5k_env_key not in os.environ:
    raise tools.UserException("Key %r not found in environment; are you running on Grid5000?" % _g5k_env_key)
  multi = pathlib.Path(os.environ[_g5k_env_key]).read_text().strip().split(os.linesep)
  seens = set()
  nodes = []
  for node in multi:
    if node in seens:
      continue
    nodes.append(node + ":7000")
    seens.add(node)
  _g5k_cluster = {"ps": nodes[0:1], "workers": nodes[1:]}
  return _g5k_cluster

# ---------------------------------------------------------------------------- #
# Main cluster parser helper

_cluster_parsers = {
  "G5k": _g5k_parser }

# String representing the list of supported special parsers
cluster_parsers = ("', '").join(_cluster_parsers.keys())
if len(cluster_parsers) > 0:
  cluster_parsers = "'" + cluster_parsers + "'"

def cluster_parse(text):
  """ Parse the given cluster representation.
  Args:
    text Cluster JSON representation or a special parser name
  Returns:
    Cluster dictionary
  """
  global _cluster_parsers
  if text in _cluster_parsers:
    return _cluster_parsers[text]()
  return json.loads(text)
