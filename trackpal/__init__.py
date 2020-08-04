"""#TrackPal: Tracking Python AnaLyzer

A modular library for the analysis of 2D tracks in Python with pandas.

## Overview

Tracks are represented as a pandas DataFrame with a minium of 4 columns

## Example:

```python
import pandas as pd
import trackpal as tp


trj_brownian = tp.simulate.brownian(n_tracks=10)
trj_linear = tp.simulate.brownian_linear(n_tracks=10)

trj_brownian["label"] = 0
trj_linear["label"] = 1

# concatenate and relabel
trj = tp.concat_relabel([trj_linear, trj_brownian])

# plot as lines
trj.groupby(trj.trackid).apply(
    tp.visu.plot_trj, coords=trj.coords, line_fmt=".-",
)

# prepare feature factory
feature_factory = tp.features.Features(frame=trj.frameid, coords=trj.coords)

# compute two features
conf_ratio = feature_factory.get("confinement_ratio")
speed_stats = feature_factory.get("speed_stats")

conf_ratio_res = trj.groupby(trj.trackid).apply(conf_ratio.compute)
speed_stats_res = trj.groupby(trj.trackid).apply(speed_stats.compute)

# retrieve labels assignment
y = trj.groupby(trj.trackid)["label"].first()

# merge into single DataFrame
features = pd.concat([conf_ratio_res, speed_stats_res, y], axis=1)

# plot with pandas
features.plot.scatter(
    x="confinement_ratio", y="speed_stats_mean", c="label", cmap="coolwarm"
)



```



##


## Submodules:

* `trackpal.read`: reading from comma separated text files (.csv) from Imaris and from (.xml) TrackMate

* `trackpal.simulate`: simulate brownian, linear and saltatory motion

* `trackpal.features`: compute feature descriptors for tracks for subsequent analysis

* `trackpal.visu`: visualize tracks

* `trackpal.drift`: correct for drift

* `trackpal.msd`:
    mean squared displacement curves

* `trackpal.msd`:
    velocity autocorrelation curves


Source code and issue tracker: [on github](https://www.github.com/sommerc/trackpal)


"""

__all__ = [
    "drift",
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

from .utils import clone_meta_attr
from . import drift, features, fit, msd, read, velocity, version, visu, simulate


@clone_meta_attr
def concat_relabel(trj_list, trackid=None):
    if hasattr(trj_list[0], "trackid"):
        trackid = getattr(trj_list[0], "trackid")
    elif trackid is None:
        from .simulate import trackid

    cnt = count(start=0, step=1)
    res = []
    for i, trj in enumerate(trj_list):
        for t_id in trj[trackid].unique():
            cur_trj = trj.loc[trj[trackid] == t_id].copy()
            cur_trj[trackid] = next(cnt)
            res.append(cur_trj)
    return pd.concat(res, axis=0, ignore_index=True).reset_index(drop=True)

