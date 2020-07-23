import numpy

np = numpy
import pandas

pd = pandas
import pathlib
from xml.etree import cElementTree as ET


def trackmate_xml_tracks(fn):
    """Reads tracks from trackmate xml track file and returns a DataFrame
       plus other additional data info

    Args:
        [str] fn (file name): [description]

    Returns:
        [type]: [description]
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
    """Reads tracks from images csv file and returns a DataFrame"""
    data = pandas.read_csv(fn, skiprows=3, sep=",",)
    data = data[data.TrackID.notna()]
    cols = ["Position X", "Position Y", "Position Z", "Time"]

    for c in cols:
        data[c] = pandas.to_numeric(data[c])
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


def imaris_tracks_custom(in_dir):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        root = pathlib.Path(in_dir)
        all_data = []
        trackid = 0
        for d in root.glob("**/*"):
            if d.is_file() and d.name.endswith(".csv"):
                treatment = d.stem.split("_")[-1]
                if treatment == "Control":
                    treatment = treatment.lower()

                print("Reading file", d.name)
                data = imaris_tracks(d.absolute())

                data["Group"] = treatment
                data["Slice"] = d.stem

                for tid in data["TrackID"].unique():

                    cur_data = data.loc[data["TrackID"] == tid]
                    cur_data["TrackID"] = trackid
                    trackid += 1

                    all_data.append(cur_data)

    return pandas.concat(all_data, axis=0)
