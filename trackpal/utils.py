import pandas
import numpy; np = numpy
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def plot_trj(trj, coords,  ax=None,
                           scale=10,
                           line_fmt='x:',
                           line_color="gray",
                           line_label="Trajectory",
                           line_width=None,
                           marker_size=None,
                           alpha=None,
                           start_end=(True, True)):
    if not ax:
        ax = plt.gca()
    ax.plot(*(trj[coords].values.T), line_fmt, color=line_color, label=line_label, lw=line_width, markersize=marker_size, alpha=alpha)

    if start_end[0]:
        ax.plot(*trj[coords].iloc[0].T, 'o', color='lightgreen')

    if start_end[1]:
        ax.plot(*trj[coords].iloc[-1].T, 'o', color='red')

    ax.axis('off')

    if scale is not None:
        ax.plot([trj[coords[0]].mean()-scale/2, trj[coords[0]].mean()+scale/2], [trj[coords[1]].min()-3, trj[coords[1]].min()-3] , 'k-', lw=3)

    ax.set_aspect(1.)




def abline(intercept, slope, col="k", label=""):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, col+'--', label=label)

def parabola(t, D, V):
    return D*t + V*(t**2)

def fit_parabola(x, y, clip=0.25):
    if 0<clip<=1:
        clip_int = int(len(x) * clip)
    else:
        clip_int = int(clip)


    x = x[:clip_int]
    y = y[:clip_int]

    (D, V2), cov = curve_fit(parabola, x, y, p0=(1,1), bounds=[(-numpy.inf,-numpy.inf), (numpy.inf, numpy.inf)])

    residuals = y - parabola(x, D, V2)
    ss_res = numpy.sum(residuals**2)
    ss_tot = numpy.sum((y-numpy.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)


    return D, V2, r2


def fit_line(taus, msd, sem, clip_first=0.25):
    if 0<clip_first<1:
        clip_int = int(len(taus) * clip_first)-1
    else:
        clip_int = int(clip_int)

    # clip data for fit to only use first part
    X =      taus[:clip_int]
    Y =  msd[:clip_int]
    W = 1/sem[:clip_int]

    # weighted LS
    X = sm.add_constant(X)
    wls_model = sm.WLS(Y, X, weights=W)
    fit_params = wls_model.fit().params
    return fit_params


frame = 'FRAME'
coords = ['Position X', 'Position Y']
trackid = 'TrackID'

def make_demo_tracks():
    res = []
    n_tracks = 20
    max_time = 100
    rand_spread = 10
    max_xy = 1000
    t_id = 0

    drift_func = lambda t: numpy.c_[-t, -t]

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