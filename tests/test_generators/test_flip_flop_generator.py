"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

import mlrose_ky
from mlrose_ky import FlipFlopOpt
from mlrose_ky.generators import FlipFlopGenerator


# noinspection PyTypeChecker
class TestFlipFlopGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero."""
        with pytest.raises(ValueError) as excinfo:
            FlipFlopGenerator.generate(SEED, size=0)
        assert str(excinfo.value) == "Size must be a positive integer. Got 0."

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is negative."""
        with pytest.raises(ValueError) as excinfo:
            FlipFlopGenerator.generate(SEED, size=-5)
        assert str(excinfo.value) == "Size must be a positive integer. Got -5."

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value."""
        with pytest.raises(TypeError):
            FlipFlopGenerator.generate(SEED, size="ten")

    def test_generate_default_size(self):
        """Test generate method with default size."""
        problem = FlipFlopGenerator.generate(SEED)

        assert problem.length == 20

    def test_generate_with_seed(self):
        """Test generate method with a specified SEED."""
        problem = FlipFlopGenerator.generate(SEED)
        np.random.seed(SEED)
        expected_problem = FlipFlopOpt(length=20)

        assert problem.length == expected_problem.length
        assert problem.__class__ == expected_problem.__class__

    def test_generate_custom_size(self):
        """Test generate method with custom size."""
        size = 30
        problem = FlipFlopGenerator.generate(SEED, size=size)

        assert problem.length == size
