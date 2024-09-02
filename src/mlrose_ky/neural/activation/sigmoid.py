"""Neural network activation functions."""

import warnings

import numpy as np

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause
from mlrose_ky.decorators import short_name

warnings.filterwarnings("ignore")


@short_name("sigmoid")
def sigmoid(x, deriv=False):
    """Sigmoid activation function

    Parameters
    ----------
    x: np.ndarray
        Array containing input data.

    deriv: bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    fx: np.ndarray
        Value of activation function at x
    """
    fx = 1 / (1 + np.exp(-x))

    if deriv:
        fx *= 1 - fx

    return fx
