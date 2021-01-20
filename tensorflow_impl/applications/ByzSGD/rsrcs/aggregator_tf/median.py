from pathlib import Path
import tensorflow as tf
import numpy as np


class Median:

    @staticmethod
    def aggregate(gradients, native=False):
        assert len(gradients) > 0, "Empty list of gradient to aggregate"

        if native:
            base_path = Path(__file__).parent.parent
            lib = tf.load_op_library(str(base_path) + '/native/op_median.so')
            return lib.Median(gradients=gradients)

        if len(gradients) > 1:
            return np.median(gradients, axis=0)
        else:
            return gradients[0]
