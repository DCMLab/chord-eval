#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fractions import Fraction

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chord_eval.metric import get_distance

PITCH_TO_STRING = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


def overlap(i1: float, i2: float) -> bool:
    """
    Indicates if the two intervals overlap

    Parameters
    ----------
    i1 : float
            first interval : array of two elements.
    i2 : float
            second interval : array of two elements.

    Returns
    -------
    bool
            True if the two intervals overlap.

    """
    if i1[1] > i2[0] and i1[1] <= i2[1]:
        return True
    if i2[1] > i1[0] and i2[1] <= i1[1]:
        return True
    return False


def duration_overlap(i1: float, i2: float) -> float:
    """
    Return the duration of the common section of the two intervals.

    Parameters
    ----------
    i1 : float
            first interval : array of two elements.
    i2 : float
            second interval : array of two elements.

    Returns
    -------
    float
            duration of the common section of the two intervals.

    """
    if i1[1] > i2[0] and i1[1] <= i2[1]:
        if i1[0] < i2[0]:
            return i1[1] - i2[0]
        else:
            return i1[1] - i1[0]
    if i2[1] > i1[0] and i2[1] <= i1[1]:
        if i2[0] < i1[0]:
            return i2[1] - i1[0]
        else:
            return i2[1] - i2[0]
    return 0


def test_columns(df: pd.DataFrame):
    """
    Indicates if the DataFrame contains the needed columns for the get_progression
    function

    Parameters
    ----------
    df : pd.DataFrame
            A DataFrame passed in argument for the get_progression function.

    Raises
    ------
    KeyError
            If some needed columns are missing a KeyError is raised.

    Returns
    -------
    None.

    """
    columns_df = df.columns
    if "chord_root_midi" not in columns_df:
        raise KeyError("'chord_root_midi' must be a column of the data frame.")

    if "chord_type" not in columns_df:
        raise KeyError("'chord_type' must be a column of the data frame.")

    if "chord_inversion" not in columns_df:
        raise KeyError("'chord_inversion' must be a column of the data frame.")

    if "duration" not in columns_df:
        raise KeyError("'duration' must be a column of the data frame.")

    if "chord_suspension_midi" not in columns_df:
        raise KeyError("'chord_suspension_midi' must be a column of the data frame.")

    df["duration"] = df["duration"].apply(lambda r: Fraction(r))


def get_progression(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    triad_reduction: bool = False,
    sps_kws: dict = {},
    vl_kws: dict = {},
    tbt_kws: dict = {},
) -> pd.DataFrame:
    """
    Construct a pd.DataFrame where each row is a event : a chord change in one of
    the annotation.


    Parameters
    ----------
    df1 : pd.DataFrame
            The first annotation of the piece.

    df2 : pd.DataFrame
            The second annotation of the piece.

    triad_reduction : bool, optional
    If True, the function will reduce every input chords to their triad
    reduction.
    The default is False.

    sps_kws: Dict
            Dictionary of keyword arguments for SPS_distance(). The defautl is {}.

    vl_kws: Dict
            Dictionary of keyword arguments for voice_leading_distance(). The defautl is {}.

    tbt_kws: Dict
            Dictionary of keyword arguments for tone_by_tone_distance(). The defautl is {}.


    Returns
    -------
    progression : pd.DataFrame
            The data frame containing the progression of the piece with the two annotations
            and the distances between each annotation..

    """

    # Test if the two DataFrames contains the needed columns for the evaluation
    test_columns(df1)
    test_columns(df2)

    # Make df2 dcml
    swap = False
    if "label" in df1.columns:
        swap = True
        df1, df2 = df2, df1

    # Creation of a new column containing the full description of each encontered
    # chord in the annotations
    df1_full_chord = df1.apply(
        lambda r: PITCH_TO_STRING[r.chord_root_midi]
        + "_"
        + str(r.chord_type).split(".")[1]
        + "_inv"
        + str(r.chord_inversion),
        axis=1,
    )
    df2_full_chord = df2.apply(
        lambda r: PITCH_TO_STRING[r.chord_root_midi]
        + "_"
        + str(r.chord_type).split(".")[1]
        + "_inv"
        + str(r.chord_inversion),
        axis=1,
    )

    # Creation of a new column containing the duration interval of each encontered
    # chord in the annotations in terme. The intervals are expressed in terms of
    # whole notes and with respect of the begining of the piece.
    time_df1 = df1.duration.cumsum().astype(float, copy=False)
    df1_interval = [[i, f] for i, f in zip([0] + list(time_df1[:-1]), time_df1)]

    time_df2 = df2.duration.cumsum().astype(float, copy=False)
    df2_interval = [[i, f] for i, f in zip([0] + list(time_df2[:-1]), time_df2)]

    # The columns of the output DataFrame
    sps = []
    vl = []
    tbt = []
    binary = []
    dcml_label = []

    time = []
    durations = []

    chords_df2 = []
    chords_df1 = []

    # For every encontered chord of one annotation, we find wich chord in the
    # other annotation it matches with
    idx_df1 = 0
    for idx_df2, rdf2 in df2.iterrows():

        matched_idx = []
        matched_duration = []

        chords_sps_dist = []
        chords_vl_dist = []
        chords_tbt_dist = []
        chords_bin_dist = []

        if idx_df1 > 0 and overlap(df1_interval[idx_df1 - 1], df2_interval[idx_df2]):

            matched_idx.append(idx_df1 - 1)

            matched_duration.append(
                duration_overlap(df1_interval[idx_df1 - 1], df2_interval[idx_df2])
            )

            chords_sps_dist.append(
                get_distance(
                    distance="SPS",
                    root1=df1.chord_root_midi[idx_df1 - 1],
                    root2=rdf2.chord_root_midi,
                    chord_type1=df1.chord_type[idx_df1 - 1],
                    chord_type2=rdf2.chord_type,
                    inversion1=df1.chord_inversion[idx_df1 - 1],
                    inversion2=rdf2.chord_inversion,
                    changes1=df1.chord_suspension_midi[idx_df1 - 1],
                    changes2=rdf2.chord_suspension_midi,
                    triad_reduction=triad_reduction,
                    **sps_kws
                )
            )

            chords_vl_dist.append(
                get_distance(
                    distance="voice leading",
                    root1=df1.chord_root_midi[idx_df1 - 1],
                    root2=rdf2.chord_root_midi,
                    chord_type1=df1.chord_type[idx_df1 - 1],
                    chord_type2=rdf2.chord_type,
                    inversion1=df1.chord_inversion[idx_df1 - 1],
                    inversion2=rdf2.chord_inversion,
                    changes1=df1.chord_suspension_midi[idx_df1 - 1],
                    changes2=rdf2.chord_suspension_midi,
                    triad_reduction=triad_reduction,
                    **vl_kws
                )
            )

            chords_tbt_dist.append(
                get_distance(
                    distance="tone by tone",
                    root1=df1.chord_root_midi[idx_df1 - 1],
                    root2=rdf2.chord_root_midi,
                    chord_type1=df1.chord_type[idx_df1 - 1],
                    chord_type2=rdf2.chord_type,
                    inversion1=df1.chord_inversion[idx_df1 - 1],
                    inversion2=rdf2.chord_inversion,
                    changes1=df1.chord_suspension_midi[idx_df1 - 1],
                    changes2=rdf2.chord_suspension_midi,
                    triad_reduction=triad_reduction,
                    **tbt_kws
                )
            )

            chords_bin_dist.append(
                0 if df1_full_chord[idx_df1 - 1] == df2_full_chord[idx_df2] else 1
            )
            dcml_label.append(rdf2.label)

        while idx_df1 < len(df1) and overlap(
            df1_interval[idx_df1], df2_interval[idx_df2]
        ):

            matched_idx.append(idx_df1)

            matched_duration.append(
                duration_overlap(df1_interval[idx_df1], df2_interval[idx_df2])
            )

            chords_sps_dist.append(
                get_distance(
                    distance="SPS",
                    root1=df1.chord_root_midi[idx_df1],
                    root2=rdf2.chord_root_midi,
                    chord_type1=df1.chord_type[idx_df1],
                    chord_type2=rdf2.chord_type,
                    inversion1=df1.chord_inversion[idx_df1],
                    inversion2=rdf2.chord_inversion,
                    changes1=df1.chord_suspension_midi[idx_df1],
                    changes2=rdf2.chord_suspension_midi,
                    triad_reduction=triad_reduction,
                    **sps_kws
                )
            )

            chords_vl_dist.append(
                get_distance(
                    distance="voice leading",
                    root1=df1.chord_root_midi[idx_df1],
                    root2=rdf2.chord_root_midi,
                    chord_type1=df1.chord_type[idx_df1],
                    chord_type2=rdf2.chord_type,
                    inversion1=df1.chord_inversion[idx_df1],
                    inversion2=rdf2.chord_inversion,
                    changes1=df1.chord_suspension_midi[idx_df1],
                    changes2=rdf2.chord_suspension_midi,
                    triad_reduction=triad_reduction,
                    **vl_kws
                )
            )

            chords_tbt_dist.append(
                get_distance(
                    distance="tone by tone",
                    root1=df1.chord_root_midi[idx_df1],
                    root2=rdf2.chord_root_midi,
                    chord_type1=df1.chord_type[idx_df1],
                    chord_type2=rdf2.chord_type,
                    inversion1=df1.chord_inversion[idx_df1],
                    inversion2=rdf2.chord_inversion,
                    changes1=df1.chord_suspension_midi[idx_df1],
                    changes2=rdf2.chord_suspension_midi,
                    triad_reduction=triad_reduction,
                    **tbt_kws
                )
            )

            chords_bin_dist.append(
                0 if df1_full_chord[idx_df1] == df2_full_chord[idx_df2] else 1
            )
            dcml_label.append(rdf2.label)

            idx_df1 += 1

        sps += chords_sps_dist
        vl += chords_vl_dist
        tbt += chords_tbt_dist
        binary += chords_bin_dist

        if len(matched_idx) > 0:

            time.append(df2_interval[idx_df2][0])
            if len(matched_idx) > 1:
                for duration in matched_duration:
                    time.append(df2_interval[idx_df2][0] + duration)
                del time[-1]

            chords_df2 += [df2_full_chord[idx_df2]] * len(matched_idx)
            chords_df1 += [df1_full_chord[idx_df1] for idx_df1 in matched_idx]

        durations += matched_duration

    progression = pd.DataFrame(
        {
            "time": time,
            "matched_duration": durations,
            "annotation1_chord": chords_df2 if swap else chords_df1,
            "annotation2_chord": chords_df1 if swap else chords_df2,
            "dcml_label": dcml_label,
            "sps": sps,
            "vl": vl,
            "tbt": tbt,
            "binary": binary,
        }
    )
    return progression


def plot_comparison(
    progression: pd.DataFrame,
    figsize: float = (22, 10),
    rge: float = None,
    verbose: bool = False,
    title: str = None,
    annotation1: str = None,
    annotation2: str = None,
):
    """
    Plot the progression of 2 annotaions

    Parameters
    ----------
    progression : pd.DataFrame
            pd.DataFrame containg progressino the two annotaion and their distance for
            each chord.
    figsize:
            Specifies the size of the figure. The default is (22,10)
    rge : float, optional
            the range in terms of whole note of the progression : a 2-element array.
            The default is None.
    verbose : bool, optional
            if True the name of each chord of the two annotation will be marked under
            the dot of the event. The default is False.
    title : str, optional
            Title of the plot. The default is None.
    annotation1 : str, optional
            Name of the first annotation. The default is None.
    annotation2 : str, optional
            Name of the second annotation. The default is None.

    Returns
    -------
    None.

    """
    if rge is None:
        rge = [progression.time[0], progression.time.iloc[-1]]

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    sns.scatterplot(
        data=progression.query("time>=@rge[0] and time<@rge[1]"),
        x="time",
        y="sps",
        ax=axs[0],
        label="SPS",
    )
    sns.scatterplot(
        data=progression.query("time>=@rge[0] and time<@rge[1]"),
        x="time",
        y="tbt",
        ax=axs[0],
        label="tone by tone",
    )
    axs[0].grid()
    axs[0].set(xlabel="time", ylabel="distance to matched chord")

    sns.scatterplot(
        data=progression.query("time>=@rge[0] and time<@rge[1]"),
        x="time",
        y="vl",
        ax=axs[1],
        label="voice leading",
    )
    axs[1].grid()
    axs[1].set(xlabel="time", ylabel="distance to matched chord")

    if verbose:

        limit1 = progression.query("time>=@rge[0] and time<@rge[1]").index[0]
        limit2 = progression.query("time>=@rge[0] and time<@rge[1]").index[-1]

        plt.text(
            progression.loc[limit1].time,
            progression.loc[limit1].vl - 0.2,
            progression.loc[limit1].annotation2_chord,
            horizontalalignment="center",
            verticalalignment="top",
            size="small",
            fontstretch="normal",
            color="maroon",
        )
        for line in range(limit1 + 1, limit2):
            if (
                progression.annotation2_chord[line]
                != progression.annotation2_chord[line - 1]
            ):
                plt.text(
                    progression.time[line],
                    progression.vl[line] - 0.2,
                    progression.annotation2_chord[line],
                    horizontalalignment="center",
                    verticalalignment="top",
                    size="small",
                    fontstretch="normal",
                    color="maroon",
                )

        plt.text(
            progression.loc[limit1].time,
            progression.loc[limit1].vl + 0.2,
            progression.loc[limit1].annotation1_chord,
            horizontalalignment="center",
            verticalalignment="bottom",
            size="small",
            fontstretch="normal",
        )
        for line in range(limit1 + 1, limit2):
            if (
                progression.annotation1_chord[line]
                != progression.annotation1_chord[line - 1]
            ):
                plt.text(
                    progression.time[line],
                    progression.vl[line] + 0.2,
                    progression.annotation1_chord[line],
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    size="small",
                    fontstretch="normal",
                )

    if title is not None:
        fig.suptitle(title)

    if annotation1 is not None and annotation2 is not None:
        legend = annotation1 + ": upper chord\n" + annotation2 + ": lower chord"
        fig.text(0.45, -0.02, legend)
    fig.tight_layout()
