"""Velocity auto-correlation
"""

import numpy as np
import pandas as pd

from .utils import defdict2array
from functools import partial
from collections import defaultdict


def displacement(trajectory, coords, frame="FRAME"):
    """computes velocity of tracks"""
    velo = (trajectory[coords] - trajectory[coords].shift(1)).dropna()
    velo[frame] = trajectory[frame].shift(1).dropna()
    velo["dt"] = (trajectory[frame] - trajectory[frame].shift(1)).dropna()

    return velo


def autocorr_pre_track(trajectory, coords, frame="FRAME"):
    """Compute autocorrelation velocity for each trajectory
    and integrate results into a auto_corr_values dictionary.
    Note, this function has no ouput, adds key,values
    to an existent dictionary
    """
    auto_corr_values = defaultdict(list)
    n_shifts = len(trajectory)
    corr_0 = np.square(trajectory[coords]).sum(axis=1).mean()

    for shift in range(n_shifts):
        corr = trajectory[coords] * trajectory[coords].shift(shift)
        corr = corr.dropna().sum(axis=1) / corr_0

        taus = (trajectory[frame].shift(-shift) - trajectory[frame]).dropna()

        for t, m in zip(taus, corr):
            auto_corr_values[shift].append(m)

    return defdict2array(auto_corr_values)


def _velocity_autocorr_all_tracks(trajectory, auto_corr_values, coords, frame="FRAME"):
    """Compute autocorrelation velocity for each trajectory
    and integrate results into a auto_corr_values dictionary.
    Note, this function has no ouput, adds key,values
    to an existent dictionary
    """
    n_shifts = len(trajectory)
    corr_0 = np.square(trajectory[coords]).sum(axis=1).mean()

    for shift in range(n_shifts):
        corr = trajectory[coords] * trajectory[coords].shift(shift)
        corr = corr.dropna().sum(axis=1) / corr_0

        taus = (trajectory[frame].shift(-shift) - trajectory[frame]).dropna()

        for t, m in zip(taus, corr):
            auto_corr_values[shift].append(m)


def auto_correlation_curve(table_tracks, coords, trackid, frame_interval=1):
    autocorr_values = defaultdict(list)
    autocorr_values[0].append(1)  # delay has vac of 1

    velocities = table_tracks.groupby(trackid).apply(displacement, coords=coords)

    velocities.groupby(trackid).apply(
        _velocity_autocorr_all_tracks, auto_corr_values=autocorr_values, coords=coords
    )

    max_delay = int(max(autocorr_values.keys())) + 1

    ntracks = np.zeros(max_delay)
    ac_means = np.zeros(max_delay)
    ac_std = np.zeros(max_delay)

    for k, v_lst in autocorr_values.items():
        ki = int(k)
        ntracks[ki] = len(v_lst)
        ac_means[ki] = np.mean(v_lst)
        ac_std[ki] = np.std(v_lst)

    ac_std[0] = ac_std[1]  # avoid infinity weight
    ntracks[0] = ntracks[1]

    ac_sems = ac_std / np.sqrt(ntracks)

    taus = np.arange(max_delay) * frame_interval

    return pd.DataFrame({"tau": taus, "mean": ac_means, "std": ac_std, "sem": ac_sems})

