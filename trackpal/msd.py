"""Mean square displacement
"""

import numpy as np
import pandas as pd

import warnings
from functools import partial
from collections import defaultdict
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


from .utils import defdict2array


def per_track(trajectory, coords, frame="FRAME"):
    """Compute MSD for one trajectory """

    msds_values = defaultdict(list)
    n_shifts = len(trajectory)
    for shift in range(1, n_shifts):
        diffs = trajectory[coords].shift(-shift) - trajectory[coords]
        msd = np.square(diffs.dropna()).sum(axis=1)

        taus = (trajectory[frame].shift(-shift) - trajectory[frame]).dropna()

        for t, m in zip(taus, msd):
            msds_values[t].append(m)

    return defdict2array(msds_values)


def _msd_all_tracks(trajectory, msds_values, coords, frame="FRAME"):
    """Computes MSD and integrate results into a msds_values
    dictionary. Note, this function has no ouput, adds the
    pairs key,values to an existent dictionary
    """

    n_shifts = len(trajectory)
    for shift in range(1, n_shifts):
        diffs = trajectory[coords].shift(-shift) - trajectory[coords]
        msd = np.square(diffs.dropna()).sum(axis=1)

        taus = (trajectory[frame].shift(-shift) - trajectory[frame]).dropna()

        for t, m in zip(taus, msd):
            msds_values[t].append(m)


def curve(table_tracks, coords, trackid, frame_interval=1):
    msds_values = defaultdict(list)
    msds_values[0].append(0)  # delay 0 has msd of 0

    table_tracks.groupby(trackid).apply(
        _msd_all_tracks, msds_values=msds_values, coords=coords
    )

    # get mean msd and respective std
    max_delay = int(max(msds_values.keys())) + 1

    ntracks = np.zeros(max_delay)
    msds_means = np.zeros(max_delay)
    msds_std = np.zeros(max_delay)

    for k, v_lst in msds_values.items():
        ki = int(k)
        ntracks[ki] = len(v_lst)
        msds_means[ki] = np.mean(v_lst)
        msds_std[ki] = np.std(v_lst)

    msds_std[0] = msds_std[1]  # avoid infinity weight
    ntracks[0] = ntracks[1]

    msds_sems = msds_std / np.sqrt(ntracks)

    taus = np.arange(max_delay) * frame_interval
    return pd.DataFrame(
        {"tau": taus, "mean": msds_means, "std": msds_std, "sem": msds_sems}
    )

