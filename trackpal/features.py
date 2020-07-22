import numpy
import pandas
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

from . import msd, utils, velocity

class TrajectoryFeature(object):
    def __init__(self, coords, frame):
        self.coords = coords
        self.frame = frame


class MSDParabola(TrajectoryFeature):
    name = "MSD_Parabola"
    def compute(self, trj, min_trj_len=3, clip=0.9):
        tau, msd_vals = msd.msd_per_track(trj, self.coords, self.frame)
        d, v2, r2 = 0, 0, 0
        if len(tau) > min_trj_len:
            d, v2, r2 = utils.fit_parabola(tau, msd_vals, clip=clip)

        return pandas.Series({
                              "MSDparabola_d" : d,
                              "MSDparabola_v2": v2,
                              "MSDparabola_r2": r2
                              })

class VACstats(TrajectoryFeature):
    name = "VAC_stats"
    def compute(self, trj, min_trj_len=3):
        displacements = velocity.compute_velocities(trj, self.coords, self.frame)
        tau, vac_vals = velocity.velocity_autocorr_pre_track(displacements, self.coords, self.frame)

        if len(tau) > min_trj_len:
            return pandas.Series({"vac_mean": vac_vals[1:].mean(),
                                  "vac_std": vac_vals[1:].std(),
                                  "vac_min": vac_vals[2:].min(),
                                  "vac_max": vac_vals[2:].max()})
        else:
            return pandas.Series({"vac_mean": 0, "vac_std": 0, "vac_min": 0, "vac_max": 0})


class SpeedStats(TrajectoryFeature):
    name = "Speed_{}"
    def compute(self, trj, min_trj_len=2):
        displacements = velocity.compute_velocities(trj, self.coords, self.frame)

        speeds = numpy.linalg.norm(displacements[self.coords], axis=1) / trj[self.frame].diff().dropna()


        if len(speeds) >= min_trj_len:
            speed_std = speeds.std()
        else:
            speed_std = -1

        return pandas.Series({"Speed_mean": speeds.mean(),
                              "Speed_std": speed_std,
                              "Speed_min": speeds.min(),
                              "Speed_max": speeds.max()})

class ConfinementRatio(TrajectoryFeature):
    name = "ConfinementRatio"
    def compute(self, trj):
        return pandas.Series({"Track_confinment_ratio": numpy.linalg.norm(trj.iloc[-1][self.coords]
                                                                    - trj.iloc[0][self.coords])
                                                                    / numpy.linalg.norm(trj[self.coords].diff().dropna(), axis=1).sum()})

class MeanStraightLineSpeed(TrajectoryFeature):
    name = "MeanStraightLineSpeed"
    def compute(self, trj):
        return pandas.Series({"Track_mean_straight_line_speed": numpy.linalg.norm(trj.iloc[0][self.coords]
                                                                               - trj.iloc[-1][self.coords])
                                                                            / (trj.iloc[-1][self.frame]
                                                                               - trj.iloc[0][self.frame])})

class EllipseFit(TrajectoryFeature):
    name = "Gyration_tensor"
    def compute(self, trj):
        p = trj[self.coords].values
        pc = p - p.mean(0)

        cov = numpy.cov(pc.T)
        evals, evecs = numpy.linalg.eig(cov)

        sort_indices = numpy.argsort(evals)

        l1 = numpy.sqrt( evals[sort_indices[0]])
        l2 = numpy.sqrt( evals[sort_indices[1]])

        return pandas.Series({"Gyration_tensor_minor_axis_length" : l1,
                              "Gyration_tensor_major_axis_length" : l2,
                              "Gyration_tensor_axis_ratio"        : l1/(l2+10e-9),
                              "Gyration_tensor_radius"            : l1**2 + l2**2,
                              "Gyration_tensor_coherence"         : ((l2-l1) / (l2+l1))**2
                              }
                              )

class TrackDuration(TrajectoryFeature):
    name = "Track_duration"
    def compute(self, trj):
        return pandas.Series({"Track_duration": (1 + trj.iloc[-1][self.frame] - trj.iloc[0][self.frame])})

class TrackLength(TrajectoryFeature):
    name = "Track_length"
    def compute(self, trj):
        return pandas.Series({"Track_length": numpy.linalg.norm(trj[self.coords].diff().dropna(), axis=1).sum()})

class TrackDiameter(TrajectoryFeature):
    name = "track_diameter_stats"
    def compute(self, trj):
        pw = pairwise_distances(trj[self.coords].values)
        sf = squareform(pw, checks=False)

        return pandas.Series({"Track_diameter_mean"  : sf.mean(),
                              "Track_diameter_max"   : sf.max(),
                              "Track_diameter_std"   : sf.std(),
                             })

def angles_from_displacements(disp_xy):
    a = []
    eps = 10e-12
    for i in range(len(disp_xy)-1):
        v0 = disp_xy[i, :]
        v1 = disp_xy[i+1, :]
        c = (v0 @ v1) / (numpy.linalg.norm(v0) * numpy.linalg.norm(v1) + eps)
        ang = numpy.arccos(numpy.clip(c, -1, 1))

        a.append(ang)
    return a

class DirectionalChange(TrajectoryFeature):
    name = "directional_change"
    def compute(self, trj):
        disp_xy = trj[self.coords].diff().dropna().values

        a = angles_from_displacements(disp_xy)

        if len(a) > 1:
            p1 = numpy.percentile(a, 25)
            p2 = numpy.percentile(a, 50)
            p3 = numpy.percentile(a, 75)
        else:
            p1 = 0
            p2 = 0
            p3 = 0

        return pandas.Series({"Track_directional_change_mean" : numpy.mean(a),
                              "Track_directional_change_std"  : numpy.std(a),
                              "Track_directional_change_25_percentile"   : p1,
                              "Track_directional_change_median"   : p2,
                              "Track_directional_change_75_percentile"   : p3,
                             })

class DirectionalChangeCount(TrajectoryFeature):
    name = "directional_change_count"
    def compute(self, trj):
        disp_xy = trj[self.coords].diff().dropna().values

        a = angles_from_displacements(disp_xy)

        res = pandas.Series({"Track_directional_change_greater_{}_degree".format(d) : (numpy.array(a) > numpy.deg2rad(d)).sum() / len(a) for d in [20,45,90]})
        return res



class PartitionFeature(TrajectoryFeature):
    name = "Moving_tracklet"
    def compute(self, trj, eps=6, t_scale=1.):

        p = trj[self.coords].values
        t = trj[self.frame].values
        dt = numpy.diff(t)

        labels = partition_dbscann(p, t, eps, t_scale=t_scale, ) + 1

        from skimage.measure import label
        from scipy.ndimage.morphology import binary_dilation
        from skimage.morphology import remove_small_objects

        sub_track_indecies = []
        unique_labels = set(labels)
        for u in unique_labels:
            if u > 0:
                sub_track_indecies.append(numpy.nonzero(labels == u)[0])

        sub_track_center_idx  = [int(sti.mean()) for sti in sub_track_indecies]
        sub_track_center_pos  = [p[c, :] for c in sub_track_center_idx]
        sub_track_center_time = [t[c] for c in sub_track_center_idx]

        if len(unique_labels) > 2:
            between_times = numpy.diff(sub_track_center_time)
            between_dists = numpy.linalg.norm(numpy.diff(sub_track_center_pos, axis=0), axis=1)

        else:
            between_times = 0
            between_dists = 0

        res = {
                "Dwell_state_count" : labels.max(),
                "Dwell_state_total_duration" : (dt * (labels[1:] > 0)).sum(),
                "Dwell_state_relative_duration" : (dt * (labels[1:] > 0)).sum() / dt.sum(),

                "Dwell_between_state_duration_mean" : numpy.mean(between_times),
                "Dwell_between_state_duration_max"  : numpy.max(between_times),
                "Dwell_between_state_length_mean" : numpy.mean(between_dists),
                "Dwell_between_state_length_max"  : numpy.max(between_dists),
                }

        labels = labels==0
        labels = remove_small_objects(labels, 3)
        labels = label(labels)


        sub_track_indecies = []
        unique_labels = set(labels)
        for u in unique_labels:
            if u > 0:
                sub_track_indecies.append(numpy.nonzero(labels == u)[0])

        sub_track_pos  = [p[sti, :] for sti in sub_track_indecies]
        sub_track_time = [t[sti] for sti in sub_track_indecies]

        sub_trajectoris = [pandas.DataFrame({self.coords[0] : p[:,0], self.coords[1] : p[:,1], self.frame:t}) for p,t in zip(sub_track_pos, sub_track_time)]

        sub_trajectoris = [st for st in sub_trajectoris if len(st > 3)]

        if len(sub_trajectoris) == 0:
            sub_trajectoris = [trj]

        SS = SpeedStats(self.coords, self.frame)
        speed_vals = pandas.concat([SS.compute(st) for st in sub_trajectoris], axis=1)
        for key, vals in speed_vals.iterrows():
            res[f"{self.name}__{key}__mean"] = vals.mean()
            # res[f"{self.name}__{key}__std"]  = vals.std()



        ELF = EllipseFit(self.coords, self.frame)
        elf_vals = pandas.concat([ELF.compute(st) for st in sub_trajectoris], axis=1)

        for key, vals in elf_vals.iterrows():
            res[f"{self.name}__{key}__mean"] = vals.mean()
            # res[f"{self.name}__{key}__std"]  = vals.std()


        # Directonal change
        dc_vals = numpy.array([numpy.mean(angles_from_displacements(st[self.coords].diff().dropna().values))
                                                for st in sub_trajectoris])

        res[f"{self.name}__Directional_change_mean__mean"] = dc_vals.mean()
        # res[f"{self.name}__Directional_change_mean__std"]  = dc_vals.std()



        return pandas.Series(res)


def partition_dbscann(p, t, eps, t_scale=1.):
    X = numpy.c_[p, t * t_scale]
    clustering = DBSCAN(eps).fit(X)
    labels = clustering.labels_

    return labels


def partition_get_moving_part(trj, eps, t_scale, coords, frame):

    p = trj[coords].values
    t = trj[frame].values
    dt = numpy.diff(t)

    labels = partition_dbscann(p, t, eps, t_scale=t_scale, ) + 1

    from skimage.measure import label
    from scipy.ndimage.morphology import binary_dilation
    from skimage.morphology import remove_small_objects



    labels = labels==0
    labels = remove_small_objects(labels, 3)

    new_trj = trj[labels]
    new_trj[frame] = list(range(len(new_trj)))

    if len(new_trj) == 0:
        print("aASDFASDFASDF")

    return new_trj[coords + [frame]]







