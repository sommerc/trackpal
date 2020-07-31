"""TrackPal: Tracking Python AnaLyzer

Analze 2D object trackings with Python and pandas.

Main features:

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

import numpy as np
import pandas as pd

from itertools import count


def concat_relabel(trj_list, trackid=None):
    if trackid is None:
        from .simulate import trackid

    cnt = count(start=0, step=1)
    res = []
    for i, trj in enumerate(trj_list):
        for t_id in trj[trackid].unique():
            cur_trj = trj.loc[trj[trackid] == t_id].copy()
            cur_trj[trackid] = next(cnt)
            res.append(cur_trj)
    return pd.concat(res, axis=0, ignore_index=True).reset_index(drop=True)

