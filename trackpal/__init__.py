"""# TrackPal: Tracking Python AnaLyzer

A modular library for the analysis of object trackings in Python with pandas.

## Installation

[on github](https://git.ist.ac.at/csommer/trackpal/)

## Overview

### Track representation

Tracks are represented as a (stacked) pandas DataFrame with a minium of 5 columns for:

* TrackID
* FrameID
* Position for X and Y coordinates
* TimeID

Each track must have an unique TrackID.

More columns can be added to store additional information.

A loaded or a simulated data set contains an attribute `id` which holds the
identifiers to access the columns.

The default identifiers are:

* TrackID: `"track"`
* FrameID: `"frame"`
* Position for X and Y coordinates: `"xy"`
* TimeID `"time"`

### General
For most computations trackpal relies on pandas `groupby` and `apply` mechanism.

`TrackPal` does not track or link objects. It analyzes already tracked objects.
For obtaining object trackings from images or detections see for instance the
excellent projects [TrackMate](https://imagej.net/TrackMate),
[trackpy](http://soft-matter.github.io/trackpy) or [ilastik](ilastik.org)

## Examples:

### Simulate tracks
The following types of tracks are supported:

* brownian motion (Gaussian random walk)
* linear motion
* mixed brownian and linear
* saltatory motion

Examples:
```python
import trackpal as tp

trj = tp.simulate.brownian_linear(n_tracks=10)

# plot as lines
trj.groupby(trj.trackid).apply(
    tp.visu.plot_trj, coords=trj.xy, line_fmt=".-",
)
```

Output:
![](https://git.ist.ac.at/csommer/trackpal/-/raw/master/doc/img/bl_tracks_01.png "Output")

### Track features

* Simulate different motion types and compute track feautres
    * [notebook](https://git.ist.ac.at/csommer/trackpal/-/blob/master/examples/01_track_features.ipynb)


## Source code and issue tracker:
[on github](https://git.ist.ac.at/csommer/trackpal/)


"""

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

