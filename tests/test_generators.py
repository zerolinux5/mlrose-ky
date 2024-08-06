"""Unit tests for generators/"""

# Author: Kyle Nakamura
# License: BSD 3 clause

import numpy as np

from mlrose_hiive import ContinuousPeaks, DiscreteOpt

try:
    import mlrose_hiive
except ImportError:
    import sys

    sys.path.append("..")

from mlrose_hiive.generators import (ContinuousPeaksGenerator, FlipFlopGenerator, KnapsackGenerator,
                                     MaxKColorGenerator, QueensGenerator, TSPGenerator)


class TestContinuousPeaksGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero"""
        try:
            ContinuousPeaksGenerator.generate(seed=1, size=0)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got 0"
        else:
            assert False, "ValueError not raised"

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is a negative integer"""
        try:
            ContinuousPeaksGenerator.generate(seed=1, size=-5)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got -5"
        else:
            assert False, "ValueError not raised"

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value"""
        try:
            # noinspection PyTypeChecker
            ContinuousPeaksGenerator.generate(seed=1, size="ten")
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got ten"
        else:
            assert False, "ValueError not raised"

    def test_generate_negative_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is less than 0"""
        try:
            ContinuousPeaksGenerator.generate(seed=1, size=10, threshold_percentage=-0.1)
        except ValueError as e:
            assert str(e) == "Threshold percentage must be between 0 and 1. Got -0.1"
        else:
            assert False, "ValueError not raised"

    def test_generate_threshold_percentage_greater_than_one(self):
        """Test generate method raises ValueError when threshold_percentage is greater than 1"""
        try:
            ContinuousPeaksGenerator.generate(seed=1, size=10, threshold_percentage=1.1)
        except ValueError as e:
            assert str(e) == "Threshold percentage must be between 0 and 1. Got 1.1"
        else:
            assert False, "ValueError not raised"

    def test_generate_non_float_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is a non-float value"""
        try:
            # noinspection PyTypeChecker
            ContinuousPeaksGenerator.generate(seed=1, size=10, threshold_percentage="high")
        except ValueError as e:
            assert str(e) == "Threshold percentage must be a float. Got str"
        else:
            assert False, "ValueError not raised"

    def test_generate_default_size(self):
        """Test generate method with default size"""
        seed = 42
        problem = ContinuousPeaksGenerator.generate(seed)
        assert problem.length == 20

    def test_generate_default_threshold_percentage(self):
        """Test generate method with default threshold_percentage"""
        seed = 42
        size = 20
        problem = ContinuousPeaksGenerator.generate(seed, size)
        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == 0.1

    def test_generate_with_seed(self):
        """Test generate method with a specified seed"""
        seed = 42
        problem = ContinuousPeaksGenerator.generate(seed)
        np.random.seed(seed)
        expected_fitness = ContinuousPeaks()
        expected_problem = DiscreteOpt(length=20, fitness_fn=expected_fitness)

        assert problem.length == expected_problem.length
        assert problem.fitness_fn.__class__ == expected_problem.fitness_fn.__class__
        assert problem.fitness_fn.threshold_percentage == expected_problem.fitness_fn.threshold_percentage

    def test_generate_custom_size_and_threshold(self):
        """Test generate method with custom size and threshold_percentage"""
        seed = 42
        size = 30
        threshold_percentage = 0.2
        problem = ContinuousPeaksGenerator.generate(seed, size=size, threshold_percentage=threshold_percentage)
        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == threshold_percentage
