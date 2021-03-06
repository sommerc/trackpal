"""Simulate 2D tracks with various motion types
"""

import numpy as np
import pandas as pd

from .utils import IdDefaults

coords = IdDefaults.xy
trackid = IdDefaults.track
frameid = IdDefaults.frame


def _brownian_xy(n, diffusion=1, xs_rng=(0, 100), ys_rng=(0, 100), frame_interval=1):
    x_off = np.random.rand() * (xs_rng[1] - xs_rng[0]) + xs_rng[0]
    y_off = np.random.rand() * (ys_rng[1] - ys_rng[0]) + ys_rng[1]

    pos_xy = np.random.randn(n, 2) * np.sqrt(2 * diffusion * frame_interval)
    pos_xy = np.cumsum(pos_xy, axis=0) + [y_off, x_off]

    return pos_xy


def brownian(
    n_tracks=20,
    min_time=0,
    max_time=42,
    diffusion=1,
    xs_rng=(0, 100),
    ys_rng=(0, 100),
    frame_interval=1,
):
    """Simmulates brownian motion by Gaussian random walk.

    Args:
        n_tracks (int, optional):  Defaults to 20.
        min_time (int, optional):  Defaults to 0.
        max_time (int, optional):  Defaults to 42.
        diffusion (int, optional):  Defaults to 1.
        xs_rng (tuple, optional):  Defaults to (0, 100).
        ys_rng (tuple, optional):  Defaults to (0, 100).
        frame_interval (int, optional):  Defaults to 1.

    Returns:
        pandas.DataFrame: tracks
    """

    res = []
    for t_id in range(n_tracks):

        frames = np.arange(min_time, max_time + 1, dtype=np.int32) * frame_interval
        track_ids = np.ones(len(frames), dtype=np.int32) * t_id

        pos_xy = _brownian_xy(
            len(frames),
            diffusion=diffusion,
            xs_rng=xs_rng,
            ys_rng=ys_rng,
            frame_interval=frame_interval,
        )

        df = pd.DataFrame(pos_xy, columns=coords)
        df.insert(0, frameid, frames)
        df.insert(0, trackid, track_ids)

        res.append(df)

    brownian_track = pd.concat(res, axis=0).reset_index(drop=True)

    return brownian_track


def _linear_xy(
    n, velocity=1, xs_rng=(0, 100), ys_rng=(0, 100),
):
    x_off = np.random.rand() * (xs_rng[1] - xs_rng[0]) + xs_rng[0]
    y_off = np.random.rand() * (ys_rng[1] - ys_rng[0]) + ys_rng[1]

    xy_direction = np.random.randn(1, 2)
    xy_direction = xy_direction / np.linalg.norm(xy_direction) * velocity

    pos_xy = np.tile(xy_direction, (n, 1))
    pos_xy = np.cumsum(pos_xy, axis=0) + [y_off, x_off]

    return pos_xy


def linear(
    n_tracks=20,
    min_time=0,
    max_time=42,
    velocity=1,
    xs_rng=(0, 100),
    ys_rng=(0, 100),
    frame_interval=1,
):
    """Simulate linear motion

    Args:
        n_tracks (int, optional): Defaults to 20.
        min_time (int, optional): Defaults to 0.
        max_time (int, optional): Defaults to 42.
        velocity (int, optional): Defaults to 1.
        xs_rng (tuple, optional): Defaults to (0, 100).
        ys_rng (tuple, optional): Defaults to (0, 100).
        frame_interval (int, optional): Defaults to 1.

    Returns:
        pandas.DataFrame: tracks
    """

    res = []
    for t_id in range(n_tracks):
        frames = np.arange(min_time, max_time + 1, dtype=np.int32) * frame_interval
        track_ids = np.ones(len(frames), dtype=np.int32) * t_id

        xy_direction = np.random.randn(1, 2)
        xy_direction = xy_direction / np.linalg.norm(xy_direction) * velocity

        pos_xy = _linear_xy(
            len(frames), velocity=velocity, xs_rng=xs_rng, ys_rng=ys_rng
        )

        df = pd.DataFrame(pos_xy, columns=coords)
        df.insert(0, frameid, frames)
        df.insert(0, trackid, track_ids)

        res.append(df)

    brownian_track = pd.concat(res, axis=0).reset_index(drop=True)

    return brownian_track


def brownian_linear(diffusion=1, velocity=1, **kwargs):
    """Simulate mixed linear and brownian motion

    Args:
        diffusion (int, optional):  Defaults to 1.
        velocity (int, optional):  Defaults to 1.

    Returns:
        pandas.DataFrame: tracks
    """
    b = brownian(diffusion=diffusion, **kwargs)
    l = linear(velocity=velocity, xs_rng=(0, 0), ys_rng=(0, 0), **kwargs)

    trj = b

    trj[coords] += l[coords]

    return trj


def saltatory(
    n_tracks,
    n_pauses=5,
    diffusion_pause=0.1,
    diffusion_moving=0.05,
    velocity_pause=0,
    velocity_moving=1,
    lengths=(20, 10),
    xs_rng=(0, 100),
    ys_rng=(0, 100),
    frame_interval=1,
):
    """Simulate tracks with saltatory motion

    Args:
        n_tracks (int):
        n_pauses (int, optional):  Defaults to 5.
        diffusion_pause (float, optional):  Defaults to 0.1.
        diffusion_moving (float, optional):  Defaults to 0.05.
        velocity_pause (int, optional):  Defaults to 0.
        velocity_moving (int, optional):  Defaults to 1.
        lengths (tuple, optional):  Defaults to (20, 10).
        xs_rng (tuple, optional):  Defaults to (0, 100).
        ys_rng (tuple, optional):  Defaults to (0, 100).
        frame_interval (int, optional):  Defaults to 1.

    Returns:
        pandas.DataFrame: tracks
    """
    res = []
    for t_id in range(n_tracks):
        p1 = _brownian_xy(
            lengths[0], diffusion=diffusion_pause, xs_rng=xs_rng, ys_rng=ys_rng
        )

        p2 = _linear_xy(
            lengths[0], velocity=velocity_pause, xs_rng=(0, 0), ys_rng=(0, 0)
        )

        p = p1 + p2

        cur_track = [p]
        for p in range(n_pauses - 1):

            l1 = _linear_xy(lengths[1], velocity_moving, xs_rng=(0, 0), ys_rng=(0, 0))
            l2 = _brownian_xy(
                lengths[1], diffusion=diffusion_moving, xs_rng=(0, 0), ys_rng=(0, 0)
            )
            l = l1 + l2

            l += cur_track[-1][-1, :]
            cur_track.append(l)

            p1 = _brownian_xy(
                lengths[0], diffusion=diffusion_pause, xs_rng=(0, 0), ys_rng=(0, 0)
            )
            p2 = _linear_xy(
                lengths[0], velocity=velocity_pause, xs_rng=(0, 0), ys_rng=(0, 0)
            )

            p = p1 + p2

            p += cur_track[-1][-1, :]
            cur_track.append(p)

        pos_xy = np.concatenate(cur_track, axis=0)
        frames = np.arange(0, len(pos_xy), dtype=np.int32) * frame_interval
        track_ids = np.ones(len(frames), dtype=np.int32) * t_id

        df = pd.DataFrame(pos_xy, columns=coords)
        df.insert(0, frameid, frames)
        df.insert(0, trackid, track_ids)

        res.append(df)

    return pd.concat(res, axis=0).reset_index(drop=True)

