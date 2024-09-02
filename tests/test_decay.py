"""Unit tests for neural/decay/"""

# Authors: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

try:
    import mlrose_ky
except ImportError:
    import sys

    sys.path.append("..")
    import mlrose_ky

from mlrose_ky import GeomDecay, ArithDecay, ExpDecay, CustomSchedule


def test_geom_above_min():
    """Test geometric decay evaluation function for case where result is above the minimum"""
    schedule = GeomDecay(initial_temperature=10, decay_rate=0.95, minimum_temperature=1)
    x = schedule.evaluate(5)
    assert round(x, 5) == 7.73781


def test_geom_below_min():
    """Test geometric decay evaluation function for case where result is below the minimum"""
    schedule = GeomDecay(initial_temperature=10, decay_rate=0.95, minimum_temperature=1)
    x = schedule.evaluate(50)
    assert x == 1


def test_arith_above_min():
    """Test arithmetic decay evaluation function for case where result is above the minimum"""
    schedule = ArithDecay(initial_temperature=10, decay_rate=0.95, minimum_temperature=1)
    x = schedule.evaluate(5)
    assert x == 5.25


def test_arith_below_min():
    """Test arithmetic decay evaluation function for case where result is below the minimum"""
    schedule = ArithDecay(initial_temperature=10, decay_rate=0.95, minimum_temperature=1)
    x = schedule.evaluate(50)
    assert x == 1


def test_exp_above_min():
    """Test exponential decay evaluation function for case where result is above the minimum"""
    schedule = ExpDecay(initial_temperature=10, decay_rate=0.05, minimum_temperature=1)
    x = schedule.evaluate(5)
    assert round(x, 5) == 7.78801


def test_exp_below_min():
    """Test exponential decay evaluation function for case where result is below the minimum"""
    schedule = ExpDecay(initial_temperature=10, decay_rate=0.05, minimum_temperature=1)
    x = schedule.evaluate(50)
    assert x == 1


def test_custom():
    """Test custom evaluation function"""

    # noinspection PyMissingOrEmptyDocstring
    def custom_decay_function(time: int, offset: int) -> float:
        return time + offset

    kwargs = {"offset": 10}
    schedule = CustomSchedule(custom_decay_function, **kwargs)
    x = schedule.evaluate(5)
    assert x == 15
