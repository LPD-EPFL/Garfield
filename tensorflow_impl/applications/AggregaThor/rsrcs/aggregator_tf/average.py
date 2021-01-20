import numpy as np


class Average:

    @staticmethod
    def aggregate(gradients):
        assert len(gradients) > 0, "Empty list of gradient to aggregate"

        if len(gradients) > 1:
            return np.mean(gradients, axis=0)
        else:
            return gradients[0]
