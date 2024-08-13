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
    import mlrose_hiive

from mlrose_hiive.generators import (ContinuousPeaksGenerator, FlipFlopGenerator, FourPeaksGenerator, SixPeaksGenerator,
                                     KnapsackGenerator, MaxKColorGenerator, QueensGenerator, TSPGenerator)


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


class TestFlipFlopGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero."""
        try:
            FlipFlopGenerator.generate(seed=1, size=0)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got 0."
        else:
            assert False, "ValueError not raised"

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is negative."""
        try:
            FlipFlopGenerator.generate(seed=1, size=-5)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got -5."
        else:
            assert False, "ValueError not raised"

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value."""
        try:
            # noinspection PyTypeChecker
            FlipFlopGenerator.generate(seed=1, size="ten")
        except TypeError:
            assert True  # Assuming FlipFlopOpt will raise a TypeError for non-integer size
        else:
            assert False, "TypeError not raised"

    def test_generate_default_size(self):
        """Test generate method with default size."""
        seed = 42
        problem = FlipFlopGenerator.generate(seed)

        assert problem.length == 20

    def test_generate_with_seed(self):
        """Test generate method with a specified seed."""
        seed = 42
        problem = FlipFlopGenerator.generate(seed)
        np.random.seed(seed)
        expected_problem = mlrose_hiive.FlipFlopOpt(length=20)

        assert problem.length == expected_problem.length
        assert problem.__class__ == expected_problem.__class__

    def test_generate_custom_size(self):
        """Test generate method with custom size."""
        seed = 42
        size = 30
        problem = FlipFlopGenerator.generate(seed, size=size)

        assert problem.length == size


class TestFourPeaksGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero."""
        try:
            FourPeaksGenerator.generate(seed=1, size=0)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got 0."
        else:
            assert False, "ValueError not raised"

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is negative."""
        try:
            FourPeaksGenerator.generate(seed=1, size=-5)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got -5."
        else:
            assert False, "ValueError not raised"

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value."""
        try:
            # noinspection PyTypeChecker
            FourPeaksGenerator.generate(seed=1, size="ten")
        except TypeError:
            assert True  # Assuming DiscreteOpt or FourPeaks will raise a TypeError for non-integer size
        else:
            assert False, "TypeError not raised"

    def test_generate_negative_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is less than 0."""
        try:
            FourPeaksGenerator.generate(seed=1, size=10, threshold_percentage=-0.1)
        except ValueError as e:
            assert str(e) == "Threshold percentage must be between 0 and 1. Got -0.1."
        else:
            assert False, "ValueError not raised"

    def test_generate_threshold_percentage_greater_than_one(self):
        """Test generate method raises ValueError when threshold_percentage is greater than 1."""
        try:
            FourPeaksGenerator.generate(seed=1, size=10, threshold_percentage=1.1)
        except ValueError as e:
            assert str(e) == "Threshold percentage must be between 0 and 1. Got 1.1."
        else:
            assert False, "ValueError not raised"

    def test_generate_non_float_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is a non-float value."""
        try:
            # noinspection PyTypeChecker
            FourPeaksGenerator.generate(seed=1, size=10, threshold_percentage="high")
        except TypeError:
            assert True  # Assuming FourPeaks will raise a TypeError for non-float threshold_percentage
        else:
            assert False, "TypeError not raised"

    def test_generate_default_size(self):
        """Test generate method with default size."""
        seed = 42
        problem = FourPeaksGenerator.generate(seed)

        assert problem.length == 20

    def test_generate_default_threshold_percentage(self):
        """Test generate method with default threshold_percentage."""
        seed = 42
        size = 20
        problem = FourPeaksGenerator.generate(seed, size)

        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == 0.1

    def test_generate_with_seed(self):
        """Test generate method with a specified seed."""
        seed = 42
        problem = FourPeaksGenerator.generate(seed)
        np.random.seed(seed)
        expected_fitness = mlrose_hiive.FourPeaks()
        expected_problem = DiscreteOpt(length=20, fitness_fn=expected_fitness)

        assert problem.length == expected_problem.length
        assert problem.fitness_fn.__class__ == expected_problem.fitness_fn.__class__
        assert problem.fitness_fn.threshold_percentage == expected_problem.fitness_fn.threshold_percentage

    def test_generate_custom_size_and_threshold(self):
        """Test generate method with custom size and threshold_percentage."""
        seed = 42
        size = 30
        threshold_percentage = 0.2
        problem = FourPeaksGenerator.generate(seed, size=size, threshold_percentage=threshold_percentage)

        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == threshold_percentage


class TestSixPeaksGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero."""
        try:
            SixPeaksGenerator.generate(seed=1, size=0)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got 0."
        else:
            assert False, "ValueError not raised"

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is negative."""
        try:
            SixPeaksGenerator.generate(seed=1, size=-5)
        except ValueError as e:
            assert str(e) == "Size must be a positive integer. Got -5."
        else:
            assert False, "ValueError not raised"

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value."""
        try:
            # noinspection PyTypeChecker
            SixPeaksGenerator.generate(seed=1, size="ten")
        except TypeError:
            assert True  # Assuming DiscreteOpt or SixPeaks will raise a TypeError for non-integer size
        else:
            assert False, "TypeError not raised"

    def test_generate_negative_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is less than 0."""
        try:
            SixPeaksGenerator.generate(seed=1, size=10, threshold_percentage=-0.1)
        except ValueError as e:
            assert str(e) == "Threshold percentage must be between 0 and 1. Got -0.1."
        else:
            assert False, "ValueError not raised"

    def test_generate_threshold_percentage_greater_than_one(self):
        """Test generate method raises ValueError when threshold_percentage is greater than 1."""
        try:
            SixPeaksGenerator.generate(seed=1, size=10, threshold_percentage=1.1)
        except ValueError as e:
            assert str(e) == "Threshold percentage must be between 0 and 1. Got 1.1."
        else:
            assert False, "ValueError not raised"

    def test_generate_non_float_threshold_percentage(self):
        """Test generate method raises ValueError when threshold_percentage is a non-float value."""
        try:
            # noinspection PyTypeChecker
            SixPeaksGenerator.generate(seed=1, size=10, threshold_percentage="high")
        except TypeError:
            assert True  # Assuming SixPeaks will raise a TypeError for non-float threshold_percentage
        else:
            assert False, "TypeError not raised"

    def test_generate_default_size(self):
        """Test generate method with default size."""
        seed = 42
        problem = SixPeaksGenerator.generate(seed)

        assert problem.length == 20

    def test_generate_default_threshold_percentage(self):
        """Test generate method with default threshold_percentage."""
        seed = 42
        size = 20
        problem = SixPeaksGenerator.generate(seed, size)

        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == 0.1

    def test_generate_with_seed(self):
        """Test generate method with a specified seed."""
        seed = 42
        problem = SixPeaksGenerator.generate(seed)
        np.random.seed(seed)
        expected_fitness = mlrose_hiive.SixPeaks()
        expected_problem = DiscreteOpt(length=20, fitness_fn=expected_fitness)

        assert problem.length == expected_problem.length
        assert problem.fitness_fn.__class__ == expected_problem.fitness_fn.__class__
        assert problem.fitness_fn.threshold_percentage == expected_problem.fitness_fn.threshold_percentage

    def test_generate_custom_size_and_threshold(self):
        """Test generate method with custom size and threshold_percentage."""
        seed = 42
        size = 30
        threshold_percentage = 0.2
        problem = SixPeaksGenerator.generate(seed, size=size, threshold_percentage=threshold_percentage)

        assert problem.length == size
        assert problem.fitness_fn.threshold_percentage == threshold_percentage


class TestKnapsackGenerator:

    def test_generate_invalid_seed(self):
        """Test generate method raises ValueError when seed is not an integer."""
        try:
            # noinspection PyTypeChecker
            KnapsackGenerator.generate(seed="not_an_int")
        except ValueError as e:
            assert str(e) == "Seed must be an integer. Got not_an_int"
        else:
            assert False, "ValueError not raised"

    def test_generate_invalid_number_of_item_types(self):
        """Test generate method raises ValueError when number_of_item_types is invalid."""
        try:
            KnapsackGenerator.generate(seed=1, number_of_item_types=0)
        except ValueError as e:
            assert str(e) == "Number of item types must be a positive integer. Got 0"
        else:
            assert False, "ValueError not raised"

    def test_generate_invalid_max_item_count(self):
        """Test generate method raises ValueError when max_item_count is invalid."""
        try:
            KnapsackGenerator.generate(seed=1, max_item_count=-1)
        except ValueError as e:
            assert str(e) == "Max item count must be a positive integer. Got -1"
        else:
            assert False, "ValueError not raised"

    def test_generate_invalid_max_weight_per_item(self):
        """Test generate method raises ValueError when max_weight_per_item is invalid."""
        try:
            KnapsackGenerator.generate(seed=1, max_weight_per_item=0)
        except ValueError as e:
            assert str(e) == "Max weight per item must be a positive integer. Got 0"
        else:
            assert False, "ValueError not raised"

    def test_generate_invalid_max_value_per_item(self):
        """Test generate method raises ValueError when max_value_per_item is invalid."""
        try:
            KnapsackGenerator.generate(seed=1, max_value_per_item=-10)
        except ValueError as e:
            assert str(e) == "Max value per item must be a positive integer. Got -10"
        else:
            assert False, "ValueError not raised"

    def test_generate_invalid_max_weight_percentage(self):
        """Test generate method raises ValueError when max_weight_percentage is invalid."""
        try:
            KnapsackGenerator.generate(seed=1, max_weight_percentage=1.5)
        except ValueError as e:
            assert str(e) == "Max weight percentage must be a float between 0 and 1. Got 1.5"
        else:
            assert False, "ValueError not raised"

    def test_generate_invalid_multiply_by_max_item_count(self):
        """Test generate method raises ValueError when multiply_by_max_item_count is not a boolean."""
        try:
            # noinspection PyTypeChecker
            KnapsackGenerator.generate(seed=1, multiply_by_max_item_count="yes")
        except ValueError as e:
            assert str(e) == "multiply_by_max_item_count must be a boolean. Got yes"
        else:
            assert False, "ValueError not raised"

    def test_generate_max_weight_percentage_zero(self):
        """Test generate method with max_weight_percentage set to 0"""
        seed = 42
        max_weight_percentage = 0.0

        try:
            KnapsackGenerator.generate(seed=seed, max_weight_percentage=max_weight_percentage)
        except ValueError as e:
            assert str(e) == "max_weight_pct must be greater than 0."
        else:
            assert False, "ValueError not raised"
