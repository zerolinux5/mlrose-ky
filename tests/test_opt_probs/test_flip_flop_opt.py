"""Unit tests for opt_probs/test_flip_flop_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

try:
    import mlrose_ky
except ImportError:
    import sys

    sys.path.append("../..")
    import mlrose_ky

from mlrose_ky.opt_probs import FlipFlopOpt


class TestFlipFlopOpt:
    """Tests for FlipFlopOpt class."""

    def test_set_state(self):
        """Test set_state method"""
        problem = FlipFlopOpt(5)
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        assert np.array_equal(problem.get_state(), x)

    def test_set_population(self):
        """Test set_population method"""
        problem = FlipFlopOpt(5)
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        problem.set_population(pop)
        assert np.array_equal(problem.get_population(), pop)

    def test_best_child(self):
        """Test best_child method"""
        problem = FlipFlopOpt(5)
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        problem.set_population(pop)
        x = problem.best_child()
        assert np.array_equal(x, np.array([1, 0, 1, 0, 1]))

    def test_best_neighbor(self):
        """Test best_neighbor method"""
        problem = FlipFlopOpt(5)
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        problem.neighbors = pop
        x = problem.best_neighbor()
        assert np.array_equal(x, np.array([1, 0, 1, 0, 1]))

    def test_eval_fitness(self):
        """Test eval_fitness method"""
        problem = FlipFlopOpt(5)
        x = np.array([0, 1, 0, 1, 0])
        fitness = problem.eval_fitness(x)
        assert fitness == 4
