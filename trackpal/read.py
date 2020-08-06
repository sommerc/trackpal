"""Reading tracks from Imaris (.csv) and TrackMate (.xml)
"""
import pathlib

import numpy as np
import pandas as pd

from collections import namedtuple

from xml.etree import cElementTree as ET


def trackmate_xml_tracks(fn):
    """Reads tracks from trackmate xml track file and returns a DataFrame
       plus other additional data info

    Args:
        fn (str): file name

    Returns:
        pandas.DataFrame: tracks
    """

    tracks = ET.parse(fn)
    frame_interval = float(tracks.getroot().attrib["frameInterval"])
    time_units = str(tracks.getroot().attrib["timeUnits"])
    space_units = str(tracks.getroot().attrib["spaceUnits"])

    attributes = []
    for ti, track in enumerate(tracks.iterfind("particle")):
        for spots in track.iterfind("detection"):
            attributes.append(
                [
                    ti,
                    int(spots.attrib.get("t")),
                    float(spots.attrib.get("x")),
                    float(spots.attrib.get("y")),
                ]
            )

    track_table = pd.DataFrame(
        attributes, columns=["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y"]
    )
    track_table["POSITION_T"] = track_table["FRAME"] * frame_interval

    return track_table, frame_interval, time_units, space_units


def imaris_tracks(fn):
    """Reads tracks from imars csv track file and returns a DataFrame
       plus other additional data info

    Args:
        fn (str): file name

    Returns:
        pandas.DataFrame: tracks
    """

    data = pd.read_csv(fn, skiprows=3, sep=",",)
    data = data[data.TrackID.notna()]
    cols = ["Position X", "Position Y", "Position Z", "Time"]

    for c in cols:
        data[c] = pd.to_numeric(data[c])
    data = data.sort_values(["TrackID", "Time"])

    data["FRAME"] = data["Time"]

    del data["Unnamed: 9"]
    del data["ID"]
    del data["Collection"]
    del data["Category"]
    del data["Position Z"]

    data = data.reset_index()
    del data["index"]
    return data

