# coding: utf-8
###
 # @file condense.py
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

import tools
from . import register

import math
import torch

# ---------------------------------------------------------------------------- #
# Condense random function

def aggregate(gradients, f, p=0.9, **kwargs):
  """ NaN-resilient median coordinate-per-coordinate rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    p         Median selection probability
    ...       Ignored keyword-arguments
  Returns:
    NaN-resilient, coordinate-wise median of the gradients
  """
  # Sample selection indications
  c = torch.distributions.bernoulli.Bernoulli(torch.ones_like(gradients[0]) * p).sample()
  # Compute median and mask according to selection
  m = torch.stack(gradients).median(dim=0)[0].mul_(c)
  # Inverse selection
  c.neg_().add_(1)
  # Add masked first gradient and return
  return m.add_(gradients[0].mul(c))

def check(gradients, f, p=0.9, **kwargs):
  """ Check parameter validity for the median rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    p         Median selection probability
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return "Expected a list of at least one gradient to aggregate, got %r" % (gradients,)
  if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 2:
    return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (f, (len(gradients) - 2) // 2)
  if p <= 0 or p > 1:
    return "Expected positive selection probability, got %s" % (p,)

def upper_bound(n, f, d):
  """ Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this rule.
  Args:
    n Number of workers (Byzantine + non-Byzantine)
    f Expected number of Byzantine workers
    d Dimension of the gradient space
  Returns:
    Theoretical upper-bound
  """
  return 1 / math.sqrt(n - f)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("condense", aggregate, check, upper_bound=upper_bound)
