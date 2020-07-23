import numpy as np
import pandas as pd

frame = "FRAME"
coords = ["Position X", "Position Y"]
trackid = "TrackID"


def brownian(
    n_tracks=20, min_time=0, max_time=42, diffusion=1, x_rng=(0, 100), y_rng=(0, 100)
):
    t_id = 0
    res = []
    for k_ in range(n_tracks):

        x_off = np.random.rand() * (x_rng[1] - x_rng[0]) + x_rng[0]
        y_off = np.random.rand() * (y_rng[1] - y_rng[0]) + x_rng[1]

        pos_xy = np.random.randn(max_time - min_time + 1, 2) * diffusion
        pos_xy = np.cumsum(pos_xy, axis=0) + [y_off, x_off]

        frames = np.arange(min_time, max_time + 1, dtype=np.int32)[..., None]
        track_id = np.ones((len(frames), 1), dtype=np.int32)

        df = pd.DataFrame(
            np.c_[track_id * t_id, frames, pos_xy], columns=[trackid, frame] + coords
        )

        res.append(df)
        t_id += 1

    brownian_track = pd.concat(res, axis=0)

    return brownian_track
