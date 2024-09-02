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

    def test_evaluate_population_fitness(self):
        """Test evaluate_population_fitness method"""
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
        problem.evaluate_population_fitness()
        expected_fitness = np.array([1, 4, 1, 2, 0, 0])
        assert np.array_equal(problem.get_pop_fitness(), expected_fitness)

    def test_random_pop(self):
        """Test random_pop method"""
        problem = FlipFlopOpt(5)
        problem.random_pop(10)
        pop = problem.get_population()
        assert pop.shape == (10, 5) and np.all((pop == 0) | (pop == 1))

    def test_can_stop(self):
        """Test can_stop method"""
        problem = FlipFlopOpt(5)
        x = np.array([1, 1, 1, 1, 1])
        problem.set_state(x)
        assert not problem.can_stop()
        x = np.array([1, 0, 1, 0, 1])
        problem.set_state(x)
        assert problem.can_stop()
