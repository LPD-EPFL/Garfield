import numpy as np


class Aksel:

    def __init__(self, nbbyzwrksm, mode="mid"):
        self.nbbyzwrks = nbbyzwrksm
        self.mode = mode

    def _compute_distances(self, gradients, f, mode):
        """ Aksel rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          mode      Operation mode, one of: 'mid', 'n-f'
        Returns:
          List of (gradient, distance to the median) sorted by increasing distance,
          Number of gradients to aggregate
        """
        n = len(gradients)
        # Compute median
        m = np.median(gradients, axis=0)
        # Measure squared distances to median
        d = list((i, np.linalg.norm(x - m, ord=2)**2) for i, x in enumerate(gradients))
        # Average closest to median according to mode
        if mode == "mid":
            c = (n + 1) // 2
        elif mode == "n-f":
            c = n - f
        else:
            raise NotImplementedError
        d.sort(key=lambda x: x[1])
        return d, c

    def aggregate(self, gradients):
        """ Aksel rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          mode      Operation mode, one of: 'mid', 'n-f'
          ...       Ignored keyword-arguments
        Returns:
          Aggregated gradient
        """

        check = self.check(gradients, self.nbbyzwrks, self.mode)
        if check is not None:
            raise Exception(check)

        # Compute distances and aggregate
        d, c = self._compute_distances(gradients, self.nbbyzwrks, self.mode)
        return sum(gradients[i] for i, _ in d[:c]) / c

    def check(self, gradients, f, mode="mid"):
        """ Check parameter validity for Aksel rule.
        Args:
          gradients Non-empty list of gradients to aggregate
          f         Number of Byzantine gradients to tolerate
          mode      Operation mode, see 'aggregate'
          ...       Ignored keyword-arguments
        Returns:
          None if valid, otherwise error message string
        """
        if not isinstance(gradients, list) or len(gradients) < 1:
            return "Expected a list of at least one gradient to aggregate, got %r" % gradients
        if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
            return "Invalid number of Byzantine gradients to tolerate, got f = %r, expected 1 ≤ f ≤ %d" % (
            f, (len(gradients) - 1) // 2)
        if mode not in ("mid", "n-f"):
            return f"Invalid operation mode {mode!r}"