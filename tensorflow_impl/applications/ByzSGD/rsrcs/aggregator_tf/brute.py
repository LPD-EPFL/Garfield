import math
import itertools
from .tools import pairwise
import numpy as np

class Brute:

    def __init__(self, nbbyzwrks):
        self.__nbbyzwrks = nbbyzwrks

    def _compute_selection(self, gradients, f):
        """ Brute rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
        Returns:
          Selection index set
        """

        check = self.check(gradients, f)
        if check is not None:
            raise Exception(check)

        n = len(gradients)
        # Compute all pairwise distances
        distances = [0] * (n * (n - 1) // 2)
        for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
            # distances[i] = gradients[x].sub(gradients[y]).norm().item()
            distances[i] = np.linalg.norm(gradients[x] - gradients[y], ord=2)**2

        # Select the set of smallest diameter
        sel_iset = None
        sel_diam = None
        for cur_iset in itertools.combinations(range(n), n - f):
            # Compute the current diameter (max of pairwise distances)
            cur_diam = 0.
            for x, y in pairwise(cur_iset):
                # Get distance between these two gradients ("magic" formula valid since x < y)
                cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
                # Check finite distance (non-Byzantine gradient must only contain finite coordinates), drop set if non-finite
                if not math.isfinite(cur_dist):
                    break
                # Check if new maximum
                if cur_dist > cur_diam:
                    cur_diam = cur_dist
            else:
                # Check if new selected diameter
                if sel_iset is None or cur_diam < sel_diam:
                    sel_iset = cur_iset
                    sel_diam = cur_diam
        # Return the selected gradients
        assert sel_iset is not None, "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
        return sel_iset

    def aggregate(self, gradients):
        """ Brute rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          ...       Ignored keyword-arguments
        Returns:
          Aggregated gradient
        """
        sel_iset = self._compute_selection(gradients, self.__nbbyzwrks)
        return sum(gradients[i] for i in sel_iset) / (len(gradients) - self.__nbbyzwrks)

    def check(self,gradients, f):
        """ Check parameter validity for Brute rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          ...       Ignored keyword-arguments
        Returns:
          None if valid, otherwise error message string
        """
        if not isinstance(gradients, list) or len(gradients) < 1:
            return "Expected a list of at least one gradient to aggregate, got %r" % gradients
        if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
            return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (f, (len(gradients) - 1) // 2)
