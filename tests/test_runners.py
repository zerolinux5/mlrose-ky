"""Unit tests for runners/"""

# Author: Kyle Nakamura
# License: BSD 3 clause

import os
import pytest
import signal
from unittest.mock import patch

try:
    import mlrose_ky
except ImportError:
    import sys

    sys.path.append("..")
    import mlrose_ky

# noinspection PyProtectedMember
from mlrose_ky.runners._runner_base import _RunnerBase
from mlrose_ky.runners import build_data_filename


@pytest.fixture
def test_runner():
    """Fixture to create a TestRunner instance for testing."""

    # noinspection PyMissingOrEmptyDocstring
    class TestRunner(_RunnerBase):
        def run(self):
            pass

    def _create_runner(**kwargs):
        default_kwargs = {
            "problem": None,
            "experiment_name": "test_experiment",
            "seed": 1,
            "iteration_list": [1, 2, 3],
            "output_directory": "test_output",
            "override_ctrl_c_handler": False,
        }
        # Update default_kwargs with any provided kwargs
        default_kwargs.update(kwargs)
        return TestRunner(**default_kwargs)

    return _create_runner


class TestBaseRunner:
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_increment_spawn_count(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner()
        initial_count = runner.get_spawn_count()
        runner._increment_spawn_count()

        assert runner.get_spawn_count() == initial_count + 1

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_decrement_spawn_count(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner()
        runner._increment_spawn_count()
        initial_count = runner.get_spawn_count()
        runner._decrement_spawn_count()

        assert runner.get_spawn_count() == initial_count - 1

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_get_spawn_count(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner()
        initial_spawn_count = runner.get_spawn_count()
        runner._increment_spawn_count()
        incremented_spawn_count = runner.get_spawn_count()
        assert incremented_spawn_count == initial_spawn_count + 1

        runner._decrement_spawn_count()
        decremented_spawn_count = runner.get_spawn_count()
        assert decremented_spawn_count == initial_spawn_count

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_abort_sets_abort_flag(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner()
        runner.abort()

        assert runner.has_aborted() is True

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_has_aborted_after_abort_called(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner(seed=42, iteration_list=[0])
        runner.abort()

        assert runner.has_aborted() is True

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_set_replay_mode(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner()
        assert not runner.replay_mode()

        runner.set_replay_mode()
        assert runner.replay_mode()

        runner.set_replay_mode(False)
        assert not runner.replay_mode()

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_replay_mode(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner(replay=True)
        assert runner.replay_mode() is True

        runner.set_replay_mode(False)
        assert runner.replay_mode() is False

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_setup_method(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner(problem="dummy_problem", seed=42, iteration_list=[0, 1, 2], output_directory="test_output")
        runner._setup()

        assert runner._raw_run_stats == []
        assert runner._fitness_curves == []
        assert runner._curve_base == 0
        assert runner._iteration_times == []
        assert runner._copy_zero_curve_fitness_from_first == runner._copy_zero_curve_fitness_from_first_original
        assert runner._current_logged_algorithm_args == {}
        mock_makedirs.assert_called_once_with("test_output")

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_tear_down_restores_original_sigint_handler(self, mock_exists, mock_makedirs, test_runner):
        original_handler = signal.getsignal(signal.SIGINT)
        runner = test_runner()
        runner._tear_down()
        restored_handler = signal.getsignal(signal.SIGINT)

        assert restored_handler == original_handler

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_log_current_argument(self, mock_exists, mock_makedirs, test_runner):
        runner = test_runner(seed=42, iteration_list=[0, 1, 2])
        arg_name = "test_arg"
        arg_value = "test_value"
        runner.log_current_argument(arg_name, arg_value)

        assert runner._current_logged_algorithm_args[arg_name] == arg_value


class TestRunnerUtils:
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_build_data_filename_default(self, mock_exists, mock_makedirs):
        output_directory = "test_output"
        runner_name = "TestRunner"
        experiment_name = "experiment"
        df_name = "results"

        expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results")
        result = build_data_filename(output_directory, runner_name, experiment_name, df_name)

        assert result == expected_filename
        mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_build_data_filename_with_params(self, mock_exists, mock_makedirs):
        output_directory = "test_output"
        runner_name = "TestRunner"
        experiment_name = "experiment"
        df_name = "results"
        x_param = "x_value"
        y_param = "y_value"

        expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results_x_value__y_value")
        result = build_data_filename(output_directory, runner_name, experiment_name, df_name, x_param, y_param)

        assert result == expected_filename
        mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_build_data_filename_with_extension(self, mock_exists, mock_makedirs):
        output_directory = "test_output"
        runner_name = "TestRunner"
        experiment_name = "experiment"
        df_name = "results"
        ext = "csv"

        expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results.csv")
        result = build_data_filename(output_directory, runner_name, experiment_name, df_name, ext=ext)

        assert result == expected_filename
        mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_build_data_filename_creates_directory(self, mock_exists, mock_makedirs):
        output_directory = "test_output"
        runner_name = "TestRunner"
        experiment_name = "experiment"
        df_name = "results"

        expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results")
        result = build_data_filename(output_directory, runner_name, experiment_name, df_name)

        mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)
        assert result == expected_filename
