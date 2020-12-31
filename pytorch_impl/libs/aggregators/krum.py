# coding: utf-8
###
 # @file   krum.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Multi-Krum GAR.
###

import tools
from . import register

import math
import torch

# Optional 'native' module
try:
  import native
except ImportError:
  native = None

# ---------------------------------------------------------------------------- #
# Multi-Krum GAR

def _compute_scores(gradients, f, m, **kwargs):
  """ Multi-Krum score computation.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    List of (gradient, score) by sorted (increasing) scores
  """
  n = len(gradients)
  # Compute all pairwise distances
  distances = [0] * (n * (n - 1) // 2)
  for i, (x, y) in enumerate(tools.pairwise(tuple(range(n)))):
    dist = gradients[x].sub(gradients[y]).norm().item()
    if not math.isfinite(dist):
      dist = math.inf
    distances[i] = dist
  # Compute the scores
  scores = list()
  for i in range(n):
    # Collect the distances
    grad_dists = list()
    for j in range(i):
      grad_dists.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
    for j in range(i + 1, n):
      grad_dists.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
    # Select the n - f - 1 smallest distances
    grad_dists.sort()
    scores.append((sum(grad_dists[:n - f - 1]), gradients[i]))
  # Sort the gradients by increasing scores
  scores.sort(key=lambda x: x[0])
  return scores

def aggregate(gradients, f, m=None, **kwargs):
  """ Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Defaults
  if m is None:
    m = len(gradients) - f - 2
  # Compute aggregated gradient
  scores = _compute_scores(gradients, f, m, **kwargs)
  return sum(grad for _, grad in scores[:m]).div_(m)

def aggregate_native(gradients, f, m=None, **kwargs):
  """ Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Defaults
  if m is None:
    m = len(gradients) - f - 2
  # Computation
  return native.krum.aggregate(gradients, f, m)

def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return "Expected a list of at least one gradient to aggregate, got %r" % gradients
  if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 3:
    return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (f, (len(gradients) - 3) // 2)
  if m is not None and (not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2):
    return "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d" % (m, len(gradients) - f - 2)

def upper_bound(n, f, d):
  """ Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this rule.
  Args:
    n Number of workers (Byzantine + non-Byzantine)
    f Expected number of Byzantine workers
    d Dimension of the gradient space
  Returns:
    Theoretical upper-bound
  """
  return 1 / math.sqrt(2 * (n - f + f * (n + f * (n - f - 2) - 2) / (n - 2 * f - 2)))

def influence(honests, attacks, f, m=None, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    f       Number of Byzantine gradients to tolerate
    m       Optional number of averaged gradients for Multi-Krum
    ...     Ignored keyword-arguments
  Returns:
    Ratio of accepted
  """
  gradients = honests + attacks
  # Defaults
  if m is None:
    m = len(gradients) - f - 2
  # Compute the sorted scores
  scores = _compute_scores(gradients, f, m, **kwargs)
  # Compute the influence ratio
  count = 0
  for _, gradient in scores[:m]:
    for attack in attacks:
      if gradient is attack:
        count += 1
        break
  return count / m

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "krum"
register(method_name, aggregate, check, upper_bound, influence)

# Register aggregation rule (native version, if available)
if native is not None:
  native_name = method_name
  method_name = "native-" + method_name
  if native_name in dir(native):
    register(method_name, aggregate_native, check, upper_bound)
  else:
    tools.warning("GAR %r could not be registered since the associated native module %r is unavailable" % (method_name, native_name))
