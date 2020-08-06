"""General helper and utility functions"""

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from collections import namedtuple

META_ATTR = {
    "track": "TrackID",
    "frame": "FRAME",
    "time": "TIME",
    "xy": ["Position X", "Position Y"],
}

IdAttrs = namedtuple("IDsTrackPal", META_ATTR.keys())
IdAttrs.__new__.__defaults__ = tuple(META_ATTR.values())

IdDefaults = IdAttrs()


def clone_id_attr(df_func):
    def wrapper(df_or_list, *args, **kwargs):
        if isinstance(df_or_list, (list, tuple)):
            assert len(df_or_list) > 0, "Concatenation of empty list"
            df = df_or_list[0]
        else:
            df = df_or_list

        res_df = df_func(df_or_list, *args, **kwargs)
        if hasattr(df, "id"):
            setattr(res_df, "id", getattr(df, "id"))
        return res_df

    return wrapper


def set_meta_attr(df, attr=IdDefaults):
    setattr(df, "id", attr)
    return df


def defdict2array(defdict, agg=np.mean):
    tau = np.zeros(len(defdict), dtype=int)
    arr = np.zeros(len(defdict), dtype=np.float32)
    for i, (k, v) in enumerate(sorted(defdict.items())):
        tau[i] = k
        arr[i] = agg(defdict[k])
    return tau, arr


def abline(intercept, slope, col="k", label=""):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, col + "--", label=label)


def parabola(t, D, V):
    return D * t + V * (t ** 2)


def fit_parabola(x, y, clip=0.25):
    if 0 < clip <= 1:
        clip_int = int(len(x) * clip)
    else:
        clip_int = int(clip)

    x = x[:clip_int]
    y = y[:clip_int]

    (D, V2), cov = curve_fit(
        parabola,
        x,
        y,
        p0=(1, 1),
        bounds=[(-numpy.inf, -numpy.inf), (numpy.inf, numpy.inf)],
    )

    residuals = y - parabola(x, D, V2)
    ss_res = numpy.sum(residuals ** 2)
    ss_tot = numpy.sum((y - numpy.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return D, V2, r2


def blub():
    pass


def fit_line(taus, msd, sem, clip_first=0.25):
    if 0 < clip_first < 1:
        clip_int = int(len(taus) * clip_first) - 1
    else:
        clip_int = int(clip_int)

    # clip data for fit to only use first part
    X = taus[:clip_int]
    Y = msd[:clip_int]
    W = 1 / sem[:clip_int]

    # weighted LS
    X = sm.add_constant(X)
    wls_model = sm.WLS(Y, X, weights=W)
    fit_params = wls_model.fit().params
    return fit_params

