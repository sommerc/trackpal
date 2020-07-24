import pandas as pd
import numpy as np

import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def parabola_function(t, D, V):
    return D * t + V * (t ** 2)


def parabola(x, y, clip=0.25):
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


def line(taus, msd, sem=None, clip_first=0.25):
    if 0 < clip_first < 1:
        clip_int = int(len(taus) * clip_first) - 1
    else:
        clip_int = int(clip_int)

    # clip data for fit to only use first part
    X = taus[:clip_int]
    Y = msd[:clip_int]
    if sem:
        W = 1 / sem[:clip_int]
    else:
        W = np.ones((len(X)))

    # weighted LS
    X = sm.add_constant(X)
    wls_model = sm.WLS(Y, X, weights=W)
    fit_params = wls_model.fit().params
    fit_params["diffusion_constant"] = fit_params["tau"] / 2 / 2
    return fit_params

