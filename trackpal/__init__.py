"""TrackPal: Tracking Python AnaLyzer
"""

import pathlib

# The directory containing this file
_this_dir = pathlib.Path(__file__).parent

# The text of the README file
__doc__ = (_this_dir / ".." / "README.md").read_text()


__all__ = [
    # "drift", ##TODO
    "features",
    "fit",
    "msd",
    "read",
    "simulate",
    "velocity",
    "version",
    "visu",
    "concat_relabel",
]


import numpy as np
import pandas as pd
from itertools import count

from . import features, fit, msd, read, velocity, version, visu, simulate


def concat_relabel(trj_list, trackid=None):
    """Concatenates tracks and sequentially relabels the track id

    Args:
        trj_list (list[pandas.DataFrame]): List of tracks to concatenate
        trackid (str): trackid column identifier

    Returns:
        pandas.DataFrame: Concatenated DataFrame with unique track ids
    """

    cnt = count(start=0, step=1)
    res = []
    for i, trj in enumerate(trj_list):
        for t_id in trj[trackid].unique():
            cur_trj = trj.loc[trj[trackid] == t_id].copy()
            cur_trj[trackid] = next(cnt)
            res.append(cur_trj)
    return pd.concat(res, axis=0, ignore_index=True).reset_index(drop=True)

