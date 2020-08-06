"""Helper functions for fitting lines and parabolas.

"""

import pandas as pd
import numpy as np

import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def parabola_function(t, D, V):
    """Parabola

    Args:
        t (numpy.array): [description]
        D ([type]): [description]
        V ([type]): [description]

    Returns:
        [type]: [description]
    """
    return D * t + V * (t ** 2)


def parabola(x, y, clip=0.25):
    """Fit a quadratic function

    Args:
        x (numpy.array): x-values
        y (numpy.array): y-values
        clip (float, optional): Fit only first part. Defaults to 0.25.

    Returns:
        pandas.Series: fit parameters
    """
    if 0 < clip <= 1:
        clip_int = int(len(x) * clip)
    else:
        clip_int = int(clip)

    x = x[:clip_int]
    y = y[:clip_int]

    (D, V2), cov = curve_fit(
        parabola_function,
        x,
        y,
        p0=(1, 1),
        bounds=[(-np.inf, -np.inf), (np.inf, np.inf)],
    )

    residuals = y - parabola_function(x, D, V2)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return pd.Series({"D": D, "velocity": np.sqrt(V2), "r2": r2})


def line(x, y, weights=None, clip=0.25):
    """Fit a line

    Args:
        x (numpy.array): x-values
        y (numpy.array): y-values
        clip (float, optional): Fit only first part. Defaults to 0.25.

    Returns:
        pandas.Series: fit parameters
    """
    if 0 < clip < 1:
        clip_int = int(len(x) * clip) - 1
    else:
        clip_int = int(clip_int)

    # clip data for fit to only use first part
    X = x[:clip_int]
    Y = y[:clip_int]
    if weights:
        W = 1 / weights[:clip_int]
    else:
        W = np.ones((len(X)))

    # weighted LS
    X = sm.add_constant(X)
    wls_model = sm.WLS(Y, X, weights=W)
    fit_params = wls_model.fit().params
    fit_params["diffusion_constant"] = fit_params["tau"] / 2 / 2
    return fit_params

