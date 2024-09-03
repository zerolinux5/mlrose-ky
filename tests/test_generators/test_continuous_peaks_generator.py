"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

from mlrose_ky import ContinuousPeaks, DiscreteOpt
from mlrose_ky.generators import ContinuousPeaksGenerator


# noinspection PyTypeChecker
class TestContinuousPeaksGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero"""
        with pytest.raises(ValueError) as excinfo:
            ContinuousPeaksGenerator.generate(SEED, size=0)
        assert str(excinfo.value) == "Size must be a positive integer. Got 0"

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is a negative integer"""
        with pytest.raises(ValueError) as excinfo:
            ContinuousPeaksGenerator.generate(SEED, size=-5)
        assert str(excinfo.value) == "Size must be a positive integer. Got -5"

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value"""
        with pytest.raises(ValueError) as excinfo:
            ContinuousPeaksGenerator.generate(SEED, size="ten")
        assert str(excinfo.value) == "Size must be a positive integer. Got ten"

    def test_generate_negative_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is less than 0"""
        with pytest.raises(ValueError) as excinfo:
            ContinuousPeaksGenerator.generate(SEED, size=10, threshold_percentage=-0.1)
        assert str(excinfo.value) == "Threshold percentage must be between 0 and 1. Got -0.1"

    def test_generate_threshold_percentage_greater_than_one(self):
        """Test generate method raises ValueError when threshold_percentage is greater than 1"""
        with pytest.raises(ValueError) as excinfo:
            ContinuousPeaksGenerator.generate(SEED, size=10, threshold_percentage=1.1)
        assert str(excinfo.value) == "Threshold percentage must be between 0 and 1. Got 1.1"

    def test_generate_non_float_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is a non-float value"""
        with pytest.raises(ValueError) as excinfo:
            ContinuousPeaksGenerator.generate(SEED, size=10, threshold_percentage="high")
        assert str(excinfo.value) == "Threshold percentage must be a float. Got str"

    def test_generate_default_size(self):
        """Test generate method with default size"""
        problem = ContinuousPeaksGenerator.generate(SEED)

        assert problem.length == 20

    def test_generate_default_threshold_percentage(self):
        """Test generate method with default threshold_percentage"""
        size = 20
        problem = ContinuousPeaksGenerator.generate(SEED, size)

        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == 0.1

    def test_generate_with_seed(self):
        """Test generate method with a specified SEED"""
        problem = ContinuousPeaksGenerator.generate(SEED)
        np.random.seed(SEED)
        expected_fitness = ContinuousPeaks()
        expected_problem = DiscreteOpt(length=20, fitness_fn=expected_fitness)

        assert problem.length == expected_problem.length
        assert problem.fitness_fn.__class__ == expected_problem.fitness_fn.__class__
        assert problem.fitness_fn.threshold_percentage == expected_problem.fitness_fn.threshold_percentage

    def test_generate_custom_size_and_threshold(self):
        """Test generate method with custom size and threshold_percentage"""
        size = 30
        threshold_percentage = 0.2
        problem = ContinuousPeaksGenerator.generate(SEED, size=size, threshold_percentage=threshold_percentage)

        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == threshold_percentage
