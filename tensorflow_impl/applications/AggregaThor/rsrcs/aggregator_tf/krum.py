import math
from pathlib import Path

import numpy as np
import tensorflow as tf


class Krum:

    def __init__(self, nbworkers, nbbyzwrks, native=False):
        self.__nbworkers = nbworkers
        self.__nbbyzwrks = nbbyzwrks
        self.__nbselected = nbworkers - nbbyzwrks - 2
        self.native = native

    def aggregate(self, gradients):
        # Assertion
        assert len(gradients) > 0, "Empty list of gradient to aggregate"

        check = self.check(gradients, self.__nbbyzwrks)
        if check is not None:
            raise Exception(check)

        if self.native:
            base_path = Path(__file__).parent.parent
            lib = tf.load_op_library(str(base_path) + '/native/op_krum.so')
            return lib.Krum(gradients=gradients, f=self.__nbbyzwrks, m=len(gradients) - self.__nbbyzwrks - 2)

        if self.__nbselected == self.__nbworkers:
            # Fast path average
            result = gradients[0]
            for i in range(1, self.__nbworkers):
                result += gradients[i]
            result /= float(self.__nbworkers)
            return result
        else:
            # Compute list of scores
            scores = [list() for i in range(self.__nbworkers)]
            for i in range(self.__nbworkers - 1):
                score = scores[i]
                for j in range(i + 1, self.__nbworkers):
                    # With: 0 <= i < j < nbworkers
                    #distance = deprecated_native.squared_distance(gradients[i], gradients[j])
                    distance = np.linalg.norm(gradients[i] - gradients[j], ord=2)**2
                    if math.isnan(distance):
                        distance = math.inf
                    score.append(distance)
                    scores[j].append(distance)
            nbinscore = self.__nbworkers - self.__nbbyzwrks - 2
            for i in range(self.__nbworkers):
                score = scores[i]
                score.sort()
                scores[i] = sum(score[:nbinscore])
            # Return the average of the m gradients with the smallest score
            pairs = [(gradients[i], scores[i]) for i in range(self.__nbworkers)]
            pairs.sort(key=lambda pair: pair[1])
            result = np.array(pairs[0][0])
            for i in range(1, self.__nbselected):
                result += np.array(pairs[i][0])
            result = result / float(self.__nbselected)
            return result

    def check(self, gradients, f):
        """ Check parameter validity for Multi-Krum rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
        Returns:
          None if valid, otherwise error message string
        """

        m = len(gradients) - f - 2

        if not isinstance(gradients, list) or len(gradients) < 1:
            return "Expected a list of at least one gradient to aggregate, got %r" % gradients
        if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 3:
            return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (
            f, (len(gradients) - 3) // 2)
        if m is not None and (not isinstance(m, int) or m < 1 or m > len(gradients) - f - 2):
            return "Invalid number of selected gradients, got m = %r, expected 1 ≤ m ≤ %d" % (m, len(gradients) - f - 2)