import numpy as np
from scipy.optimize import fmin
from scipy.stats import norm

from loss_calculation_functions import calculate_statistical_significance


def difference_from_sig(_audience, *args):
    """
    Calculate how much the statistical significance differs from 0.95. This function
    is used as the optimisation function in calculate_audience.
    :param _audience: Total audience across control and variant
    :param args: tuple (_conversion_rate_pc, _uplift_pc)
    :return: float
    """
    _audience = abs(_audience)
    _conversion_rate_pc, _uplift_pc = args
    return abs(0.95 - calculate_statistical_significance(
        _conversion_rate_pc, _uplift_pc, _audience))


def calculate_audience(_initial_guess, _conversion_rate_pc, _uplift):
    """
    Calculate the audience required to measure statistical significance
    :param _initial_guess: Initial guess for total audience size across both control and variant
    :param _conversion_rate_pc: % CR of control
    :param _uplift: % uplift in variant
    :return: float, warnflag of fmin function
    """
    output = fmin(
        difference_from_sig,
        _initial_guess,
        args=(_conversion_rate_pc, _uplift),
        disp=False,
        full_output=True
    )
    xmin = np.int64(abs(output[0][0]))
    warnflag = output[4]
    return xmin, warnflag


def get_initial_guess(_uplift):
    """
    Rough guess for the audience size
    :param _uplift: % uplift of the variant
    :return: float
    """
    return 3_000_000 * norm.pdf(_uplift, loc = 0, scale = 2.5)
