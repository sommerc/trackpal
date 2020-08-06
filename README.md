# TrackPal: Tracking Python AnaLyzer

A modular library for the analysis of object trackings in Python with pandas.

## Overview
### Main features
* Read Imaris / TrackMate files
* Simulate tracks with different motion types
* Compute track feature descriptors (~50 available features)
* Mean squared displacment curves for single tracks and ensembles
* Velocity autocorrelation curves for single tracks and ensembles
* Visualization utilities

For most computations trackpal relies on pandas `groupby` and `apply` mechanism.

`TrackPal` does not track or link objects. It analyzes already tracked objects.
For obtaining object trackings from images or detections see for instance the
excellent projects [TrackMate](https://imagej.net/TrackMate),
[trackpy](http://soft-matter.github.io/trackpy) or [ilastik](ilastik.org)

## Examples
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

## Installation

0. Install Anaconda Python (>=3.6) and create new environment
1. `git clone` this repostiory
2. `cd trackpal`
3. `pip install -e .`


## Documentation

Find [API documentation](https://trackpal.github.io/trackpal) on github

