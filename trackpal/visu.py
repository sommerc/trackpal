"""Track visualization"""

from matplotlib import pyplot as plt


def plot_trj(
    trj,
    coords=None,
    ax=None,
    scale=None,
    line_fmt="x:",
    line_color=None,
    line_label="Trajectory",
    line_width=None,
    marker_size=None,
    alpha=None,
    start_end=(True, True),
):
    """[summary]

    Args:
        trj (pandas.DataFrame): tracks to plot
        coords (list): The names of the x/y coodrinate column names
        ax (optional): matplotlib axes to plot in. Defaults to None.
        scale (int, optional): length of scale bar. Defaults to 10.
        line_fmt (str, optional): Defaults to "x:".
        line_color (str, optional): Defaults to "gray".
        line_label (str, optional): Defaults to "Trajectory".
        line_width ([type], optional): Defaults to None.
        marker_size ([type], optional): Defaults to None.
        alpha ([type], optional): Defaults to None.
        start_end (tuple, optional): Show marker for start/end of track. Defaults to (True, True).
    """
    if not ax:
        ax = plt.gca()

    if not coords:
        coords = trj.coords

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
