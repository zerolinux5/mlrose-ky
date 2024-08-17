import pytest
import numpy as np
import pandas as pd
import pickle as pk
import os
import hashlib
import logging
from abc import ABC

try:
    import mlrose_ky
except ImportError:
    import sys

    sys.path.append("..")
    import mlrose_ky

from mlrose_ky.runners._nn_runner_base import _NNRunnerBase
from mlrose_ky.runners._runner_base import _RunnerBase
from mlrose_ky import GridSearchMixin

SEED = 12


class TestNNRunnerBase:

    def setup_method(self):
        self.x_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(2, size=100)
        self.x_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(2, size=20)
        self.experiment_name = "test_experiment"
        self.seed = SEED
        self.iteration_list = [1, 2, 3]
        self.grid_search_parameters = {"param1": [0.1, 0.2], "param2": [1, 2]}
        self.grid_search_scorer_method = "accuracy"
        self.runner = _NNRunnerBase(
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            experiment_name=self.experiment_name,
            seed=self.seed,
            iteration_list=self.iteration_list,
            grid_search_parameters=self.grid_search_parameters,
            grid_search_scorer_method=self.grid_search_scorer_method,
        )

    def test_nn_runner_base_initialization(self):
        """Test _NNRunnerBase initialization with default parameters"""
        x_train = np.random.rand(100, 10)
        y_train = np.random.randint(2, size=100)
        x_test = np.random.rand(20, 10)
        y_test = np.random.randint(2, size=20)
        experiment_name = "test_experiment"
        seed = SEED
        iteration_list = [1, 2, 3]
        grid_search_parameters = {"param1": [0.1, 0.2], "param2": [1, 2]}
        grid_search_scorer_method = "accuracy"

        runner = _NNRunnerBase(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            experiment_name=experiment_name,
            seed=seed,
            iteration_list=iteration_list,
            grid_search_parameters=grid_search_parameters,
            grid_search_scorer_method=grid_search_scorer_method,
        )

        assert np.array_equal(runner.x_train, x_train)
        assert np.array_equal(runner.y_train, y_train)
        assert np.array_equal(runner.x_test, x_test)
        assert np.array_equal(runner.y_test, y_test)
        assert runner.experiment_name == experiment_name
        assert runner.seed == seed
        assert runner.iteration_list == iteration_list
        assert runner.grid_search_parameters == runner.build_grid_search_parameters(grid_search_parameters)
        assert runner.grid_search_scorer_method == grid_search_scorer_method
        assert runner.cv == 5
        assert runner.generate_curves is True
        assert runner.output_directory is None
        assert runner.verbose_grid_search is True
        assert runner.override_ctrl_c_handler is True
        assert runner.n_jobs == 1
        assert runner.replay is False
        assert runner.cv_results_df is None
        assert runner.best_params is None
