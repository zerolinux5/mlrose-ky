"""Unit tests for neural/fitness/network_weights.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from tests.globals import sample_data
from mlrose_ky import identity
from mlrose_ky.neural.fitness.network_weights import NetworkWeights


class TestNetworkWeights:
    """Test cases for neural network weights evaluation."""

    def test_evaluate_no_bias_classifier(self, sample_data):
        """Test evaluation of network weights without bias for classification."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.7393

    def test_evaluate_no_bias_multi(self, sample_data):
        """Test evaluation of network weights without bias for multiclass classification."""
        X, _, y_multiclass, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2]  # Unsure why last layer needs 2 nodes even though bias is False
        fitness = NetworkWeights(X, y_multiclass, node_list, activation=identity, bias=bias)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(4) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.7183

    def test_evaluate_no_bias_regressor(self, sample_data):
        """Test evaluation of network weights without bias for regression."""
        X, _, _, y_regressor = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_regressor, node_list, activation=identity, bias=bias, is_classifier=False)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.5542

    def test_evaluate_bias_regressor(self, sample_data):
        """Test evaluation of network weights with bias for regression."""
        X, _, _, y_regressor = sample_data
        hidden_nodes = [2]
        bias = True
        node_list = [5, *hidden_nodes, 1]  # Unsure why this first number needs to be 5 even though X.shape[1] is 6
        fitness = NetworkWeights(X, y_regressor, node_list, bias=bias, activation=identity, is_classifier=False)

        a = list(np.arange(10) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.4363

    def test_calculate_updates(self, sample_data):
        """Test calculation of weight updates for the network."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias, is_classifier=False, learning_rate=1)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b
        fitness.evaluate(np.asarray(weights))

        updates = list(fitness.calculate_updates())
        update1 = np.array([[-0.0017, -0.0034], [-0.0046, -0.0092], [-0.0052, -0.0104], [0.0014, 0.0028]])
        update2 = np.array([[-3.17], [-4.18]])

        assert np.allclose(updates[0], update1, atol=0.001) and np.allclose(updates[1], update2, atol=0.001)
