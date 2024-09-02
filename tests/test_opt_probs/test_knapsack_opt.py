"""Unit tests for opt_probs/knapsack_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

try:
    import mlrose_ky
except ImportError:
    import sys

    sys.path.append("../..")
    import mlrose_ky

from mlrose_ky.opt_probs import KnapsackOpt


class TestKnapsackOpt:
    """Tests for KnapsackOpt class."""

    def test_initialization(self):
        """Test initialization of KnapsackOpt"""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.5)
        assert problem.length == 3
        assert problem.max_val == 2

    def test_set_state(self):
        """Test set_state method"""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.5)
        state = [1, 0, 1]
        problem.set_state(state)
        assert problem.get_state().tolist() == state

    def test_eval_fitness(self):
        """Test eval_fitness method"""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.5)
        state = [1, 0, 1]
        fitness = problem.eval_fitness(state)
        assert fitness == 4  # Assuming the fitness function calculates correctly

    def test_set_population(self):
        """Test set_population method"""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.5)
        pop = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        problem.set_population(pop)
        assert problem.get_population().tolist() == pop

    def test_edge_cases(self):
        """Test edge cases for KnapsackOpt"""
        # Test with empty weights and values
        try:
            KnapsackOpt(weights=[], values=[], max_weight_pct=0.5)
            assert False, "Expected an exception for empty weights and values"
        except Exception as e:
            assert str(e) == "fitness_fn or both weights and values must be specified."

        # Test with invalid max_weight_pct
        try:
            KnapsackOpt(weights=[1], values=[1], max_weight_pct=-0.1)
            assert False, "Expected an exception for invalid max_weight_pct"
        except Exception as e:
            assert str(e) == "max_weight_pct must be between 0 and 1."
