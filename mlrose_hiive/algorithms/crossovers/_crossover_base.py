"""Crossover implementations for GA."""

# Author: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3 clause

from abc import ABC, abstractmethod


class _CrossOverBase(ABC):
    """Base class for crossover operations in a genetic algorithm."""

    def __init__(self, opt_prob):
        super().__init__()
        self._opt_prob = opt_prob
        self._length = opt_prob.length

    @abstractmethod
    def mate(self, p1, p2):
        pass
