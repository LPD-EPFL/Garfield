from .average import Average
from .median import Median
from .krum import Krum
from .brute import Brute
from .aksel import Aksel
from .condense import Condense
from .bulyan import Bulyan


class Aggregator_tf:

    def __init__(self, agg='Average', nb_worker=0, nb_byz_worker=0, native=False):
        _possible_aggregator = {
            'Average': Average,
            'Median': Median,
            'Krum': Krum(nb_worker, nb_byz_worker, native),
            'Brute': Brute(nb_byz_worker),
            'Aksel': Aksel(nb_byz_worker),
            'Condense': Condense(nb_byz_worker),
            'Bulyan': Bulyan(nb_byz_worker, native=native)
        }

        assert agg in _possible_aggregator.keys(), "Aggregation not implemented"

        self._aggregator = _possible_aggregator[agg]

    def aggregate(self, gradients):
        return self._aggregator.aggregate(gradients)