import math
from pathlib import Path
import tensorflow as tf

from .tools import pairwise
import numpy as np

class Bulyan:

    def __init__(self, nbbyzwrks, m=None, native=False):
        self.__nbbyzwrks = nbbyzwrks
        self.__m = m
        self.native = native

    def aggregate(self, gradients):
        """ Bulyan over Multi-Krum rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          m         Optional number of averaged gradients for Multi-Krum
        Returns:
          Aggregated gradient
        """

        n = len(gradients)
        d = gradients[0].shape[0]

        check = self.check(gradients, self.__nbbyzwrks, self.__m)
        if check is not None:
            raise Exception(check)


        if self.native:
            base_path = Path(__file__).parent.parent
            lib = tf.load_op_library(str(base_path) + '/native/op_bulyan.so')
            return lib.Bulyan(gradients=gradients, f=self.__nbbyzwrks, m=len(gradients) - self.__nbbyzwrks - 2)

        f = self.__nbbyzwrks
        m = self.__m
        # Defaults
        m_max = n - f - 2
        if m is None:
            m = m_max
        # Compute all pairwise distances
        distances = list([(math.inf, None)] * n for _ in range(n))
        for gid_x, gid_y in pairwise(tuple(range(n))):
            #dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
            dist = np.linalg.norm(gradients[gid_x] - gradients[gid_y], ord=2)**2
            if not math.isfinite(dist):
                dist = math.inf
            distances[gid_x][gid_y] = (dist, gid_y)
            distances[gid_y][gid_x] = (dist, gid_x)
        # Compute the scores
        scores = [None] * n
        for gid in range(n):
            dists = distances[gid]
            dists.sort(key=lambda x: x[0])
            dists = dists[:m]
            scores[gid] = (sum(dist for dist, _ in dists), gid)
            distances[gid] = dict(dists)
        # Selection loop
        selected = np.empty((n - 2 * f - 2, d))
        for i in range(selected.shape[0]):
            # Update 'm'
            m = min(m, m_max - i)
            # Compute the average of the selected gradients
            scores.sort(key=lambda x: x[0])
            selected[i] = sum(gradients[gid] for _, gid in scores[:m]) / m
            # Remove the gradient from the distances and scores
            gid_prune = scores[0][1]
            scores[0] = (math.inf, None)
            for score, gid in scores[1:]:
                if gid == gid_prune:
                    scores[gid] = (score - distances[gid][gid_prune], gid)
        # Coordinate-wise averaged median
        m = selected.shape[0] - 2 * f
        median = np.median(selected, axis=0)
        closests = np.argpartition(np.absolute(selected.copy() - median), -m)[-m:]  # indices of the top k elements
        closests = (closests * d) + np.arange(0, d, dtype=closests.dtype)
        avgmed = np.mean(selected[closests], axis=0)
        # Return resulting gradient
        return avgmed


    def check(self, gradients, f, m=None):
        """ Check parameter validity for Bulyan over Multi-Krum rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
        Returns:
          None if valid, otherwise error message string
        """

        if not isinstance(gradients, list) or len(gradients) < 1:
            return "Expected a list of at least one gradient to aggregate, got %r" % gradients
        if not isinstance(f, int) or f < 1 or len(gradients) < 4 * f + 3:
            return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (
            f, (len(gradients) - 3) // 4)
        if m is not None and (not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2):
            return "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d" % (f, len(gradients) - f - 2)