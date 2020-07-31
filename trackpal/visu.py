import pandas
import numpy

np = numpy
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def plot_trj(
    trj,
    coords,
    ax=None,
    scale=10,
    line_fmt="x:",
    line_color="gray",
    line_label="Trajectory",
    line_width=None,
    marker_size=None,
    alpha=None,
    start_end=(True, True),
):
    """[summary]

    Args:
        trj (DataFrame): [description]
        coords (list): The names of the x/y coodrinate column names
        ax ([type], optional): [description]. Defaults to None.
        scale (int, optional): [description]. Defaults to 10.
        line_fmt (str, optional): [description]. Defaults to "x:".
        line_color (str, optional): [description]. Defaults to "gray".
        line_label (str, optional): [description]. Defaults to "Trajectory".
        line_width ([type], optional): [description]. Defaults to None.
        marker_size ([type], optional): [description]. Defaults to None.
        alpha ([type], optional): [description]. Defaults to None.
        start_end (tuple, optional): [description]. Defaults to (True, True).
    """
    if not ax:
        ax = plt.gca()
    ax.plot(
        *(trj[coords].values.T),
        line_fmt,
        color=line_color,
        label=line_label,
        lw=line_width,
        markersize=marker_size,
        alpha=alpha
    )

    if start_end[0]:
        ax.plot(*trj[coords].iloc[0].T, "o", color="lightgreen")

    if start_end[1]:
        ax.plot(*trj[coords].iloc[-1].T, "o", color="red")

    ax.axis("off")

    if scale is not None:
        ax.plot(
            [trj[coords[0]].mean() - scale / 2, trj[coords[0]].mean() + scale / 2],
            [trj[coords[1]].min() - 3, trj[coords[1]].min() - 3],
            "k-",
            lw=3,
        )

    ax.set_aspect(1.0)
