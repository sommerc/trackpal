import numpy; np=numpy
import pandas; pd=pandas

import warnings
from functools import partial
from collections import defaultdict
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

class DriftVelocity(object):
    def __init__(self, trackid, coords, frame):
        self.coords = coords
        self.frame = frame
        self.trackid = trackid

    def estimate(self, tracks):
        offsets_at_t = defaultdict(list)

        trj_grp = tracks.groupby(self.trackid)
        for _, trj in trj_grp:
            velo = (trj[self.coords] - trj[self.coords].shift(1)).dropna()
            velo[self.frame] = trj[self.frame].shift(1).dropna()
            velo.apply(lambda xxx: offsets_at_t[xxx[self.frame]].append(xxx[self.coords].values), axis=1)

        displ_means = {int(k): numpy.array(v).mean(0) for k, v in sorted(offsets_at_t.items())}

        return displ_means


def test():
    frame = 'FRAME'
    coords = ['Position X', 'Position Y']
    trackid = 'TrackID'

    def make_demo_tracks():
        res = []
        n_tracks = 20
        max_time = 100
        rand_spread = 15
        max_xy = 1000
        t_id = 0

        drift_func = lambda t: numpy.c_[numpy.sin(t/max_time*numpy.pi*4), numpy.cos(t/max_time*numpy.pi*4)] * t

        for k_ in range(n_tracks):
            ts = numpy.random.randint(0, max_time//4)
            te = numpy.random.randint(max_time//2, max_time)

            pos_xy = numpy.random.randn(te-ts+1, 2) * (numpy.random.rand() * rand_spread +1)
            pos_xy = numpy.cumsum(pos_xy, axis=0)

            pos_xy = pos_xy + numpy.random.rand(1,2)*max_xy

            frames = numpy.arange(ts, te+1)[..., None]

            if True:
                pos_xy += drift_func(frames)

            df = pandas.DataFrame(numpy.c_[numpy.ones((len(pos_xy), 1), 'int32') * t_id, frames, pos_xy], columns=[trackid, frame] + coords)

            res.append(df)
            t_id +=1

        demo_track = pandas.concat(res, axis=0)
        demo_track['Slice'] = 'test'

        return demo_track

    demo_track = make_demo_tracks()

    D = {}
    for s, s_data in demo_track.groupby('Slice'):
        print(s)
        a = DriftVelocity('TrackID', coords, frame)
        s_drift = a.estimate(s_data)


        cum_drift = numpy.cumsum(numpy.array(list(s_drift.values())), axis=0)

        D[s] = cum_drift

        f, ax = plt.subplots(sharex=True, sharey=True, figsize=(40,40))


        utils.plot_trj(pandas.DataFrame(cum_drift, columns=coords), coords=coords, line_fmt='-', line_color='r')
        ax.set_aspect('equal')
        ax.set_title('Drift: ' + s)

        for _, trj in s_data.groupby(trackid):
            utils.plot_trj(trj, coords=coords, scale=None, ax=ax, line_fmt='.-')

        frames = numpy.arange(max_time)[..., None]
        utils.plot_trj(pandas.DataFrame(numpy.c_[frames, drift_func(frames)], columns=[frame] + coords)
                    , coords=coords, scale=None, ax=ax, line_fmt='.-', line_color='g')

        plt.savefig('Drift_'  + s + '.pdf', bbox_inches='tight')


