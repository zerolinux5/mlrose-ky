"""GA Mutators."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3 clause

from abc import ABC, abstractmethod


class _MutatorBase(ABC):
    """Base class for mutation operations in a genetic algorithm."""

    def __init__(self, opt_prob):
        super().__init__()
        self._opt_prob = opt_prob
        self._length = opt_prob.length

    @abstractmethod
    def mutate(self, child, mutation_probability):
        pass
