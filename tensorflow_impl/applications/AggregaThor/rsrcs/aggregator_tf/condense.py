import numpy as np


class Condense:

    def __init__(self, nbbyzwrks):
        self.__nbbyzwrks = nbbyzwrks

    def aggregate(self, gradients, p=0.9):
        """ NaN-resilient median coordinate-per-coordinate rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          p         Median selection probability
        Returns:
          NaN-resilient, coordinate-wise median of the gradients
        """

        f = self.__nbbyzwrks

        check = self.check(gradients, f, p)
        if check is not None:
            raise Exception(check)

        # Sample selection indications
        c = np.random.binomial(1, p, size=len(gradients[0]))
        # Compute median and mask according to selection
        m = np.median(gradients, axis=0) * c
        # Inverse selection
        c = -1 * c + 1
        # Add masked first gradient and return
        return m + (gradients[0] * c)

    def check(self, gradients, f, p=0.9):
        """ Check parameter validity for the median rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          p         Median selection probability
        Returns:
          None if valid, otherwise error message string
        """

        if not isinstance(gradients, list) or len(gradients) < 1:
            return "Expected a list of at least one gradient to aggregate, got %r" % (gradients,)
        if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 2:
            return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (
            f, (len(gradients) - 2) // 2)
        if p <= 0 or p > 1:
            return "Expected positive selection probability, got %s" % (p,)
