"""Module for track feature descriptors.
"""

import numpy as np
import pandas as pd

from rdp import rdp

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

from . import msd, utils, velocity

EPS = 10e-12


class Features:
    """Factory class for the creating of `TrackFeature`

    """

    def __init__(self, coords, frame):
        self.coords = coords
        self.frame = frame

        self._lookup = {}
        for klass in TrackFeature.__subclasses__():
            self._lookup[klass.name] = klass

    def get(self, name):
        """Retrieve track feature by name"""
        return self._lookup[name](coords=self.coords, frame=self.frame)

    @staticmethod
    def list(verbose=False):
        """List names of all available features
        """
        for klass in sorted(TrackFeature.__subclasses__(), key=lambda k: k.name):
            print(klass.name)
            if verbose:
                print(klass.compute.__doc__)


class TrackFeature:
    """Base class for all track features

    """

    def __init__(self, coords, frame):
        self.coords = coords
        self.frame = frame


class MSDParabola(TrackFeature):
    """Fits a parabola to mean squared displacment curve of track and returns
    fit parameters and \( R^2 \)
    """

    name = "msd_parabola"

    def compute(self, trj, clip=0.9, min_trj_len=3):
        tau, msd_vals = msd.per_track(trj, self.coords, self.frame)
        d, v2, r2 = 0, 0, 0
        if len(tau) > min_trj_len:
            d, v2, r2 = utils.fit_parabola(tau, msd_vals, clip=clip)

        return pd.Series(
            {f"{self.name}_a": d, f"{self.name}_b": v2, f"{self.name}_fit_r2": r2}
        )


class VACstats(TrackFeature):
    """Computes velocity autocorrelation and returns statistics

    * mean
    * std
    * min
    * max

    of the curve for delays \( > 0\)

    """

    name = "vac_stats"

    def compute(self, trj, min_trj_len=3):
        displacements = velocity.compute_velocities(trj, self.coords, self.frame)
        tau, vac_vals = velocity.velocity_autocorr_pre_track(
            displacements, self.coords, self.frame
        )

        if len(tau) > min_trj_len:
            return pd.Series(
                {
                    f"{self.name}_mean": vac_vals[1:].mean(),
                    f"{self.name}_std": vac_vals[1:].std(),
                    f"{self.name}_min": vac_vals[2:].min(),
                    f"{self.name}_max": vac_vals[2:].max(),
                }
            )
        else:
            return pd.Series(
                {
                    f"{self.name}_mean": 0,
                    f"{self.name}_std": 0,
                    f"{self.name}_min": 0,
                    f"{self.name}_max": 0,
                }
            )


class SpeedStats(TrackFeature):
    """Compute several statistics of the instantaneous speed

        * mean
        * std
        * min
        * max
    """

    name = "speed_stats"

    def compute(self, trj, min_trj_len=2):

        displacements = velocity.displacement(trj, self.coords, self.frame)

        speeds = (
            np.linalg.norm(displacements[self.coords], axis=1)
            / trj[self.frame].diff().dropna()
        )

        if len(speeds) >= min_trj_len:
            speed_std = speeds.std()
        else:
            speed_std = -1

        return pd.Series(
            {
                f"{self.name}_mean": speeds.mean(),
                f"{self.name}_std": speed_std,
                f"{self.name}_min": speeds.min(),
                f"{self.name}_max": speeds.max(),
            }
        )


class ConfinementRatio(TrackFeature):
    """Compute confinement ration as net distance divided by total distance travelled
        $$\\frac{\\|x_n-x_1\\|}{\\sum_{i=1}^n \\|x_i\\|}$$
    """

    name = "confinement_ratio"

    def compute(self, trj):
        return pd.Series(
            {
                f"{self.name}": np.linalg.norm(
                    trj.iloc[-1][self.coords] - trj.iloc[0][self.coords]
                )
                / np.linalg.norm(trj[self.coords].diff().dropna(), axis=1).sum()
            }
        )


class MeanStraightLineSpeed(TrackFeature):
    """Compute mean straight line speed as net distance divided by total time
        $$\\frac{\\|x_n-x_1\\|}{\\sum_{i=1}^n \\|\\Delta t_i\\|}$$
    """

    name = "mean_straight_line_speed"

    def compute(self, trj):
        return pd.Series(
            {
                f"{self.name}": np.linalg.norm(
                    trj.iloc[0][self.coords] - trj.iloc[-1][self.coords]
                )
                / (trj.iloc[-1][self.frame] - trj.iloc[0][self.frame])
            }
        )


class GyrationTensor(TrackFeature):
    """Compute gyration tensor by fitting an ellipse and extraction several measures

    * axis lengths
    * axis ratio
    * radius \(R^2\)
    * coherence
    """

    name = "gyration_tensor"

    def compute(self, trj):
        p = trj[self.coords].values
        pc = p - p.mean(0)

        cov = np.cov(pc.T)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)

        l1 = np.sqrt(evals[sort_indices[0]])
        l2 = np.sqrt(evals[sort_indices[1]])

        return pd.Series(
            {
                f"{self.name}_minor_axis_length": l1,
                f"{self.name}_major_axis_length": l2,
                f"{self.name}_axis_ratio": l1 / (l2 + EPS),
                f"{self.name}_radius": l1 ** 2 + l2 ** 2,
                f"{self.name}_coherence": ((l2 - l1) / (l2 + l1)) ** 2,
            }
        )


class TrackDuration(TrackFeature):
    """Duration of track
    """

    name = "track_duration"

    def compute(self, trj):
        return pd.Series(
            {f"{self.name}": (1 + trj.iloc[-1][self.frame] - trj.iloc[0][self.frame])}
        )


class TrackLength(TrackFeature):
    """Length of track
    """

    name = "track_length"

    def compute(self, trj):
        return pd.Series(
            {
                f"{self.name}": np.linalg.norm(
                    trj[self.coords].diff().dropna(), axis=1
                ).sum()
            }
        )


class VelocityAverage(TrackFeature):
    """Average velocity
    """

    name = "velocity_average"

    def compute(self, trj):
        tl = TrackLength(self.coords, self.frame).compute
        td = TrackDuration(self.coords, self.frame).compute
        return pd.Series(
            {"velocity_average": tl(trj).track_length / td(trj).track_duration}
        )


class TrackDiameter(TrackFeature):
    """Statistics of the track diameter

    * mean
    * std
    * max

    of all pairwise distances of all positions in the track
    """

    name = "track_diameter_stats"

    def compute(self, trj):
        pw = pairwise_distances(trj[self.coords].values)
        sf = squareform(pw, checks=False)

        return pd.Series(
            {
                f"{self.name}_mean": sf.mean(),
                f"{self.name}_max": sf.max(),
                f"{self.name}_std": sf.std(),
            }
        )


def angles_from_displacements(disp_xy):
    """Compute the angle in radians from track displacements

    Args:
        disp_xy (numpy.array): displacements [n x 2]

    Returns:
        numpy.array: angle in radioans [n x 1]
    """
    a = []
    for i in range(len(disp_xy) - 1):
        v0 = disp_xy[i, :]
        v1 = disp_xy[i + 1, :]
        c = (v0 @ v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + EPS)
        ang = np.arccos(np.clip(c, -1, 1))

        a.append(ang)
    return a


class RDP_simplification(TrackFeature):
    """Applies [Ramer–Douglas–Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) algorithm to simplify the track for given
    epsilon and extracts:

    * ratio of positions in simplified and original track
    * mean and std directional change
    """

    name = "rdp_simplification"

    def compute(self, trj, min_trj_len=3, eps=1):
        trj_rdp = rdp(trj[self.coords].values, epsilon=eps)
        displ_xy = np.diff(trj_rdp, axis=0)
        ang = angles_from_displacements(displ_xy)

        if len(ang) > min_trj_len:
            m = np.mean(ang)
            s = np.std(ang)
        else:
            m, s = 0, 0

        return pd.Series(
            {
                f"{self.name}_ratio_{eps}": len(trj_rdp) / len(trj),
                f"{self.name}_directional_change_mean_{eps}": m,
                f"{self.name}_directional_change_std_{eps}": s,
            }
        )


class DirectionalChange(TrackFeature):
    """Compute statistics of the angle between two successive positions
    """

    name = "directional_change_stats"

    def compute(self, trj):
        disp_xy = trj[self.coords].diff().dropna().values

        a = angles_from_displacements(disp_xy)

        if len(a) > 1:
            p1 = np.percentile(a, 25)
            p2 = np.percentile(a, 50)
            p3 = np.percentile(a, 75)
        else:
            p1 = 0
            p2 = 0
            p3 = 0

        return pd.Series(
            {
                f"{self.name}_mean": np.mean(a),
                f"{self.name}_std": np.std(a),
                f"{self.name}_25_percentile": p1,
                f"{self.name}_median": p2,
                f"{self.name}_75_percentile": p3,
            }
        )


class DirectionalChangeCount(TrackFeature):
    """Count of angles exceeding 20, 45, or 90 degree angle turns
    """

    name = "directional_change_count"

    def compute(self, trj):
        disp_xy = trj[self.coords].diff().dropna().values

        a = angles_from_displacements(disp_xy)

        res = pd.Series(
            {
                f"{self.name}_greater_{d}_degree": (np.array(a) > np.deg2rad(d)).sum()
                / len(a)
                for d in [20, 45, 90]
            }
        )
        return res


class PartitionFeature(TrackFeature):
    """Compute features based on spatio-temporal clustering with DBSCAN.

        Track is partitioned by `partition_dbscann` and features of stationary
        (dwelling) and moving tracklets are computed.

        For the stationary part:

        * partition_dwell_state_count
        * partition_dwell_state_total_duration
        * partition_dwell_state_relative_duration
        * partition_dwell_between_state_duration_mean
        * partition_dwell_between_state_duration_max
        * partition_dwell_between_state_length_mean
        * partition_dwell_between_state_length_max

        For the moving tracklets

        * partition_moving_speed_stats_mean__mean
        * partition_moving_speed_stats_std__mean
        * partition_moving_speed_stats_min__mean
        * partition_moving_speed_stats_max__mean
        * partition_moving_gyration_tensor_minor_axis_length__mean
        * partition_moving_gyration_tensor_major_axis_length__mean
        * partition_moving_gyration_tensor_axis_ratio__mean
        * partition_moving_gyration_tensor_radius__mean
        * partition_moving_gyration_tensor_coherence__mean
        * partition_moving_directional_change_mean__mean

    """

    name = "partition"

    def compute(self, trj, eps=6, t_scale=1.0):

        p = trj[self.coords].values
        t = trj[self.frame].values
        dt = np.diff(t)

        labels = partition_dbscann(p, t, eps, t_scale=t_scale,) + 1

        from skimage.measure import label
        from scipy.ndimage.morphology import binary_dilation
        from skimage.morphology import remove_small_objects

        sub_track_indecies = []
        unique_labels = set(labels)
        for u in unique_labels:
            if u > 0:
                sub_track_indecies.append(np.nonzero(labels == u)[0])

        sub_track_center_idx = [int(sti.mean()) for sti in sub_track_indecies]
        sub_track_center_pos = [p[c, :] for c in sub_track_center_idx]
        sub_track_center_time = [t[c] for c in sub_track_center_idx]

        if len(unique_labels) > 2:
            between_times = np.diff(sub_track_center_time)
            between_dists = np.linalg.norm(
                np.diff(sub_track_center_pos, axis=0), axis=1
            )

        else:
            between_times = 0
            between_dists = 0

        res = {
            f"{self.name}_dwell_state_count": labels.max(),
            f"{self.name}_dwell_state_total_duration": (dt * (labels[1:] > 0)).sum(),
            f"{self.name}_dwell_state_relative_duration": (dt * (labels[1:] > 0)).sum()
            / dt.sum(),
            f"{self.name}_dwell_between_state_duration_mean": np.mean(between_times),
            f"{self.name}_dwell_between_state_duration_max": np.max(between_times),
            f"{self.name}_dwell_between_state_length_mean": np.mean(between_dists),
            f"{self.name}_dwell_between_state_length_max": np.max(between_dists),
        }

        labels = labels == 0
        labels = remove_small_objects(labels, 3)
        labels = label(labels)

        sub_track_indecies = []
        unique_labels = set(labels)
        for u in unique_labels:
            if u > 0:
                sub_track_indecies.append(np.nonzero(labels == u)[0])

        sub_track_pos = [p[sti, :] for sti in sub_track_indecies]
        sub_track_time = [t[sti] for sti in sub_track_indecies]

        sub_trajectoris = [
            pd.DataFrame(
                {self.coords[0]: p[:, 0], self.coords[1]: p[:, 1], self.frame: t}
            )
            for p, t in zip(sub_track_pos, sub_track_time)
        ]

        sub_trajectoris = [st for st in sub_trajectoris if len(st > 3)]

        if len(sub_trajectoris) == 0:
            sub_trajectoris = [trj]

        SS = SpeedStats(self.coords, self.frame)
        speed_vals = pd.concat([SS.compute(st) for st in sub_trajectoris], axis=1)
        for key, vals in speed_vals.iterrows():
            res[f"{self.name}_moving_{key}__mean"] = vals.mean()

        ELF = GyrationTensor(self.coords, self.frame)
        elf_vals = pd.concat([ELF.compute(st) for st in sub_trajectoris], axis=1)

        for key, vals in elf_vals.iterrows():
            res[f"{self.name}_moving_{key}__mean"] = vals.mean()

        # Directonal change
        dc_vals = np.array(
            [
                np.mean(
                    angles_from_displacements(st[self.coords].diff().dropna().values)
                )
                for st in sub_trajectoris
            ]
        )

        res[f"{self.name}__directional_change_mean__mean"] = dc_vals.mean()

        return pd.Series(res)


def partition_dbscann(p, t, eps, t_scale=1.0):
    """Partition track by spatio-temporal clustering using DBSCAN

    Args:
        p (numpy.array): xy location
        t (numpy.array): times
        eps (float): DBSCAN epsilon
        t_scale (float, optional): [description]. Defaults to 1.0.

    Returns:
        numpy.array: DBSCAN label assingment
    """
    X = np.c_[p, t * t_scale]
    clustering = DBSCAN(eps).fit(X)
    labels = clustering.labels_

    return labels


def partition_get_moving_part(trj, eps, t_scale, coords, frame):
    """Helper function for visualization
    """
    p = trj[coords].values
    t = trj[frame].values
    dt = np.diff(t)

    labels = partition_dbscann(p, t, eps, t_scale=t_scale,) + 1

    from skimage.measure import label
    from scipy.ndimage.morphology import binary_dilation
    from skimage.morphology import remove_small_objects

    labels = labels == 0
    labels = remove_small_objects(labels, 3)

    new_trj = trj[labels]
    new_trj[frame] = list(range(len(new_trj)))

    return new_trj[coords + [frame]]

