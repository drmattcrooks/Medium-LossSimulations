from scipy.stats import beta
import numpy as np


def draw_samples(_audience, _conversion_rate_pc, _uplift_pc):
    """
    Draw 10000 samples from a beta distibution. The total number of trials is given by _audience.
    The successes are calculated as 0.5 * _uplift_pc * _conversion_rate_pc * _audience. We're using
    a uniform prior.
    :param _audience: total number of trials across both control and variant (int)
    :param _conversion_rate_pc: baseline success-rate (%)
    :param _uplift_pc: increase in success-rate over baseline (%)
    :return: array containing 10_000 samples
    """
    conversion_rate_pc_with_uplift = _conversion_rate_pc * (1 + _uplift_pc / 100)
    alpha_posterior = 1 + 0.5 * _audience * conversion_rate_pc_with_uplift / 100
    beta_posterior = 1 + 0.5 * _audience * (1 - conversion_rate_pc_with_uplift / 100)
    return beta.rvs(alpha_posterior, beta_posterior, size=10_000)


def risk_of_choosing_variant(_control_samples, _variant_samples):
    """
    Calculate loss associated with two sets of samples
    :param _control_samples: samples from the control posterior
    :param _variant_samples: samples from the variant posterior
    :return: float
    """
    return 100 * np.sum([x for x in _control_samples - _variant_samples if x > 0]) / np.sum(_control_samples)


def calculate_loss(
        _conversion_rate_pc,
        _uplift_pc,
        _audience
):
    """
    Calculate loss from experiment stats
    :param _conversion_rate_pc: % conversion-rate % of control
    :param _uplift_pc: % uplift in variant
    :param _audience: audience size (int)
    :return: float (%)
    """
    control_samples = draw_samples(_audience, _conversion_rate_pc, 0)
    variant_samples = draw_samples(_audience, _conversion_rate_pc, _uplift_pc)

    return risk_of_choosing_variant(control_samples, variant_samples)


def statistical_significance(_control_samples, _variant_samples, _uplift_pc):
    """
    Calculate Chance to Beat Control from posterior samples
    :param _control_samples: array of samples from control posterior
    :param _variant_samples: array of samples from variant posterior
    :param _uplift_pc: % uplift in conversion-rate in the variant
    :return: float
    """
    if _uplift_pc >= 0:
        return np.mean(_variant_samples > _control_samples)
    else:
        return np.mean(_variant_samples < _control_samples)


def calculate_statistical_significance(
        _conversion_rate_pc,
        _uplift_pc,
        _audience
):
    """
    Calculate statistical significance from experiment stats
    :param _conversion_rate_pc: % conversion-rate of control
    :param _uplift_pc: % uplift in CR of variant
    :param _audience: total samples across both control and variant
    :return: float
    """
    control_samples = draw_samples(_audience, _conversion_rate_pc, 0)
    variant_samples = draw_samples(_audience, _conversion_rate_pc, _uplift_pc)

    return statistical_significance(control_samples, variant_samples, _uplift_pc)
