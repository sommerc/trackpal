import numpy as np
import pandas as pd

frame = "FRAME"
coords = ["Position X", "Position Y"]
trackid = "TrackID"


def brownian(
    n_tracks=20, min_time=0, max_time=42, diffusion=1, x_rng=(0, 100), y_rng=(0, 100)
):

    res = []
    for t_id in range(n_tracks):

        x_off = np.random.rand() * (x_rng[1] - x_rng[0]) + x_rng[0]
        y_off = np.random.rand() * (y_rng[1] - y_rng[0]) + x_rng[1]

        pos_xy = np.random.randn(max_time - min_time + 1, 2) * diffusion
        pos_xy = np.cumsum(pos_xy, axis=0) + [y_off, x_off]

        frames = np.arange(min_time, max_time + 1, dtype=np.int32)[..., None]
        track_ids = np.ones((len(frames), 1), dtype=np.int32) * t_id

        df = pd.DataFrame(pos_xy, columns=coords)
        df.insert(0, frame, frames)
        df.insert(0, trackid, track_ids)

        res.append(df)

    brownian_track = pd.concat(res, axis=0)

    return brownian_track
