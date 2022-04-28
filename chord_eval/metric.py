#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.spatial import distance

from chord_eval.constants import TRIAD_REDUCTION
from chord_eval.data_types import ChordType, PitchType
from chord_eval.utils import (
    Distance,
    filter_noise,
    find_notes_matching,
    find_peaks,
    get_chord_pitches,
    get_dft_from_chord,
)


def SPS_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    changes1: str = None,
    changes2: str = None,
    program1: int = 0,
    program2: int = 0,
    pitches: Tuple[int] = None,
    transform: str = "vqt",
    hop_length: int = 512,
    bins_per_octave: int = 60,
    n_mels: int = 512,
    noise_filtering: bool = False,
    peak_picking: bool = False,
) -> float:
    """
    Get the spectral pitch similarity (SPS) between two chords (composed of
    a root a ChordType and an inversion) using general MIDI programs.

    Parameters
    ----------
    root1 : int
        The root of the given first chord, as MIDI note number. If the chord
        is some inversion, the root pitch will be on this MIDI note, but there
        may be other pitches below it.

    root2 : int
        The root of the given second chord, as MIDI note number. If the chord
        is some inversion, the root pitch will be on this MIDI note, but there
        may be other pitches below it.

    chord_type1 : ChordType
        The chord type of the given first chord.

    chord_type2 : ChordType
        The chord type of the given second chord.

    inversion1 : int, optional
        The inversion of the first chord.
        The default is 0.

    inversion2 : int, optional
        The inversion of the second chord.
        The default is 0.

    changes1 : str
        Any alterations to the 1st chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    changes2 : str
        Any alterations to the 2nd chord's pitches.

    pitches : Tuple[int]
        A Tuple of possible absolute MIDI pitch numbers to use in each chord.
        If given, only a subset of these pitches will be included in the generated
        chord. Specifically, those which share a pitch class with any of the
        default chord tones. Note that this means that some default chord tones might
        not be present in the generated chord.

    program1 : int, optional
        The general MIDI program number used by the fluidsynth to synthesize
        the wave form of the MIDI data of the first chord (it uses the
        TimGM6mb.sf2 sound font file included with pretty_midi and synthesizes
        at fs=44100hz by default)
        The general MIDI program number (instrument index) is in [0, 127] :
        https://pjb.com.au/muscript/gm.html
        The default is 0.

    program2 : int, optional
        The general MIDI program number used by the fluidsynth to synthesize
        the wave form of the MIDI data of the second chord.
        The default is 0.

    transform : str, optional
        Indicates which transform to use to get the spectrogram of the
        synthesized instrument's notes. It can either be 'stft, 'cqt', 'vqt'
        or 'mel'/'melspectrogram'.
        The default is 'vqt'

    hop_length : int, optional
        Number of samples between successive CQT or VQT columns if transform is
        'cqt' or 'vqt'.
        the default is 512.

    bin_per_octave : int, optional
        Number of bins per octave if transform is 'cqt' or 'vqt'.
        the default is 60.

    n_mels : int, optional
        Number of Mel bands to generate if transform is 'mel' or 'melspectrogram'.
        the default is 512.

    noise_filtering : bool, optional
        If True, the function will filter out the noise of the two spectra
        using the filter_noise function.
        If peak_picking is True, it will automatically assign the True value
        to noise_filtering.
        The default is False.

    peak_picking : bool, optional
        If True, the function will isolate the peaks of each spectrum using
        the find_peaks function after filtering out the noise of each spectrum.
                If peak_picking is True, it will automatically assign the True value
        to noise_filtering.
        The default is False.

    Returns
    -------
    dist : float
        The cosine distance beween the spectra of the two synthesized chords
        (SPS) (in [0, 1]).
    """

    dft1 = get_dft_from_chord(
        root=root1,
        chord_type=chord_type1,
        inversion=inversion1,
        changes=changes1,
        program=program1,
        pitches=pitches,
        transform=transform,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_mels=n_mels,
    )

    dft2 = get_dft_from_chord(
        root=root2,
        chord_type=chord_type2,
        inversion=inversion2,
        changes=changes2,
        pitches=pitches,
        program=program2,
        transform=transform,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_mels=n_mels,
    )

    if peak_picking or noise_filtering:
        dft1 = [filter_noise(dft) for dft in dft1]
        dft2 = [filter_noise(dft) for dft in dft2]

        if peak_picking:
            dft1 = [find_peaks(dft) for dft in dft1]
            dft2 = [find_peaks(dft) for dft in dft2]

    sps = min([distance.cosine(d1, d2) for d1, d2 in itertools.product(dft1, dft2)])

    return sps


@lru_cache
def mechanical_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    changes1: str = None,
    changes2: str = None,
    distance: Distance = Distance(),
    bass_distance: Distance = None,
    bass_weight: int = 1,
) -> float:
    """
    Get the voice leading distance between two chords : the number
    of semitones between the pitches of each chords.

    Parameters
    ----------
    root1 : int
        The root of the given first chord, as MIDI note number. If the chord
        is some inversion, the root pitch will be on this MIDI note, but there
        may be other pitches below it.

    root2 : int
        The root of the given second chord, as MIDI note number. If the chord
        is some inversion, the root pitch will be on this MIDI note, but there
        may be other pitches below it.

    chord_type1 : ChordType
        The chord type of the given first chord.

    chord_type2 : ChordType
        The chord type of the given second chord.

    inversion1 : int, optional
        The inversion of the first chord. The default is 0.

    inversion2 : int, optional
        The inversion of the second chord.. The default is 0.

    changes1 : str
        Any alterations to the 1st chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    changes2 : str
        Any alterations to the 2nd chord's pitches.

    distance : Distance
        The Distance measurement to use between notes. By default, this is
        semitone distance.

    bass_distance : Distance
        The Distance measurement to use between bass notes, if not the same as the
        distance parameter.

    bass_weight : int, optional
        The weight of the basses distance of the two chords. The default is 1.

    Returns
    -------
    total_steps : int
        The mechanical distance between the pitches of the two chords.
    """
    assert (
        distance.is_valid()
    ), "Given distance is not valid (some distances are negative)."
    if bass_distance is None:
        bass_distance = distance
    else:
        if not bass_distance.is_valid():
            logging.warning(
                "bass_distance is not valid (some values are negative). "
                "Defaulting to given distance."
            )
            bass_distance = distance

    # note number of the pitches in the chord
    notes1 = get_chord_pitches(
        root=root1,
        chord_type=chord_type1,
        pitch_type=PitchType.MIDI,
        inversion=inversion1,
        changes=changes1,
    )

    notes2 = get_chord_pitches(
        root=root2,
        chord_type=chord_type2,
        pitch_type=PitchType.MIDI,
        inversion=inversion2,
        changes=changes2,
    )

    # Bass distance
    bass_steps = bass_distance.distance_between(notes1[0], notes2[0]) * bass_weight

    # Other pitches
    upper_steps, match_ids = find_notes_matching(notes1, notes2, distance)

    # If the bass notes have been matched together, do not count again
    if (0, 0) in match_ids:
        upper_steps -= distance.distance_between(notes1[0], notes2[0])

    # Handle chords of different lengths
    if len(notes1) != len(notes2):
        big_chord, bc_idx, small_chord = (
            (notes1, 0, notes2) if len(notes1) > len(notes2) else (notes2, 1, notes1)
        )

        # Remove all pitches from the biggest chord that have already been matched
        big_chord = [
            big_chord[idx]
            for idx in range(len(big_chord))
            if idx not in set([pair[bc_idx] for pair in match_ids] + [0])
        ]

        # Find corresponding note in the smaller chord for each additional big chord note
        upper_steps += sum(
            [
                min(
                    [
                        distance.distance_between(bc_pitch, sc_pitch)
                        for sc_pitch in small_chord
                    ]
                )
                for bc_pitch in big_chord
            ]
        )

    return int(upper_steps + bass_steps)


@lru_cache
def tone_by_tone_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    changes1: str = None,
    changes2: str = None,
    root_bonus: int = 0,
    bass_bonus: int = 0,
    pitch_type: PitchType = PitchType.MIDI,
) -> float:
    """
    Get the tone by tone distance between two chords: The average of the proporion
    of pitch classes in each chord that are unmatched in the other chord.

    For example, for C major and A minor triads, 1/3 of each one's pitch classes
    are unmatched, so the tone-by-tone distance is 1/3.

    For C major and C7, 0 of the C major's pitch classes are unmatched, and 1/4 of
    the C7's pitch classes are unmatched, so the distance is 1/8.


    Parameters
    ----------
    root1 : int
        The root of the given first chord, either as a MIDI note number or as
        a TPC note number (interval above C in fifths with F = -1, C = 0, G = 1, etc.).
        If the chord is some inversion, the root pitch will be on this pitch, but there
        may be other pitches below it.

    root2 : int
        The root of the 2nd chord.

    chord_type1 : ChordType
        The chord type of the given first chord.

    chord_type2 : ChordType
        The chord type of the given second chord.

    inversion1 : int, optional
        The inversion of the first chord.
        The default is 0.

    inversion2 : int, optional
        The inversion of the second chord.
        The default is 0.

    changes1 : str
        Any alterations to the 1st chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    changes2 : str
        Any alterations to the 2nd chord's pitches.

    root_bonus : int, optional
        Give additional accuracy to chords whose root notes match (and penalize those whose
        do not). 1 will give an additional bonus or penalty of equal weighting as each other
        note.

        For example, for a C major vs C minor comparison:
            - Default distance (root_bonus == 0): 1/3 (each chord matches 2/3 of their notes
              with the other chord).
            - With root_bonus == 1: 1/4 (each chord matches 2/3 of their notes with the other
              chord, plus an additional 1 "bonus match" up to 3/4 for their roots matching).

        For example, for C major vs Cmin7 comparison:
            - Default distance: 5 / 12 (0.417 -- the average of 1/3 from the C major triad and
              2/4 from the Cmin7).
            - With root_bonus == 1: 13/40 (0.325 -- the average of 1/4 from the C major triad
              and 2/5 from the Cmin7).

        For example, for C major vs A minor:
            - Default distance: 1/3
            - With root_bonus == 1: 1/2

    bass_bonus : int, optional
        Give additional accuracy to chords whose bass notes match. 1 will give an additional
        bonus of equal weighting as each other note.

        For example, for a C major vs C minor comparison:
            - Default distance (bass_bonus == 0): 1/3 (each chord matches 2/3 of their notes
              with the other chord).
            - With bass_bonus == 1: 1/4 (each chord matches 2/3 of their notes with the other
              chord, plus an additional 1 "bonus match" up to 3/4 for their bass notes matching).

        For example, for C major vs Cmin7 comparison:
            - Default distance: 5 / 12 (0.417 -- the average of 1/3 from the C major triad and
              2/4 from the Cmin7).
            - With bass_bonus == 1: 13/40 (0.325 -- the average of 1/4 from the C major triad
              and 2/5 from the Cmin7).

        For example, for C major vs A minor:
            - Default distance: 1/3
            - With bass_bonus == 1: 1/2

    pitch_type : PitchType
        The pitch type in which root notes are encoded, and using which tone-by-tone distance
        should be calculated. With MIDI (default), enharmonic equivalence is assumed. Otherwise,
        C# and Db count as different pitches.

    Returns
    -------
    distance : float
        The tone by tone distance between two chords.
    """

    def one_sided_tbt(
        note_set1: set,
        note_set2: set,
        root_matches: bool,
        bass_matches: bool,
        root_bonus: int,
        bass_bonus: int,
    ) -> float:
        """
        Get the one-sided tbt. That is, the proportion of chord 1's notes which are missing
        from chord2, including root and bass bonus.

        Parameters
        ----------
        note_set1 : set
            The set of pitch classes contained in chord 1.
        note_set2 : np.ndarray
            The set of pitch classes in chord 2.
        root_matches : bool
            True if the chord roots match. False otherwise.
        bass_matches : bool
            True if the chord bass notes match. False otherwise.
        root_bonus : int
            The root bonus to use (see the tone_by_tone_distance function's comments for details).
        bass_bonus : int
            The bass bonus to use (see the tone_by_tone_distance function's comments for details).

        Returns
        -------
        distance : float
            The one-sided tone-by-tone distance.
        """
        matches = len(note_set1.intersection(note_set2))

        if root_matches:
            matches += root_bonus
        if bass_matches:
            matches += bass_bonus

        return 1 - matches / (len(note_set1) + bass_bonus + root_bonus)

    notes1 = get_chord_pitches(
        root=root1,
        chord_type=chord_type1,
        pitch_type=PitchType.MIDI,
        inversion=inversion1,
        changes=changes1,
    )
    if pitch_type == PitchType.MIDI:
        notes1 %= 12

    notes2 = get_chord_pitches(
        root=root2,
        chord_type=chord_type2,
        pitch_type=PitchType.MIDI,
        inversion=inversion2,
        changes=changes2,
    )
    if pitch_type == PitchType.MIDI:
        notes2 %= 12

    root_matches = root1 == root2
    bass_matches = notes1[0] == notes2[0]

    distance = np.mean(
        [
            one_sided_tbt(
                set(notes1),
                set(notes2),
                root_matches,
                bass_matches,
                root_bonus,
                bass_bonus,
            ),
            one_sided_tbt(
                set(notes2),
                set(notes1),
                root_matches,
                bass_matches,
                root_bonus,
                bass_bonus,
            ),
        ]
    )

    return distance


def get_distance(
    distance: str,
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    changes1: str = None,
    changes2: str = None,
    triad_reduction: bool = False,
    **kwargs,
) -> float:
    """
    Get the required distance between two chords : either the SPS, the voice leading,
    the tone by tone distance by calling the SPS_distance, voice_leading_distance
    or the tone_by_tone_distance function respectively, or the binary distance.
    voice.

    Parameters
    ----------
    distance : str
        The name of the metric to use. It can be either 'SPS', 'voice leading',
        'tone by tone' or 'binary'.

    root1 : int
        The root pitch of the given first chord. With distance == "tone by tone",
        this may be either a MIDI pitch class or a TPC pitch class (see the
        tone_by_tone_distance function documentation for details). Otherwise,
        thei should be a MIDI pitch class.

    root2 : int
        The root of the given second chord.

    chord_type1 : ChordType
        The chord type of the given first chord.

    chord_type2 : ChordType
        The chord type of the given second chord.

    inversion1 : int, optional
        The inversion of the first chord.
        The default is 0.

    inversion2 : int, optional
        The inversion of the second chord.
        The default is 0.

    changes1 : str
        Any alterations to the 1st chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    changes2 : str
        Any alterations to the 2nd chord's pitches.

    triad_reduction : bool, optional
        If True, the function will reduce every input chords to their triad
        reduction.
        The default is False.

    **kwargs : TYPE
        Additional argument for the type of metric used.
        If distance is 'SPS', this will be arguments for the SPS_distance function.
        If distance is 'mechanical', this will be arguments for the mechanical_distance function.
        If distance is 'tone_by_tone', this will be arguments for the
        tone_by_tone_distance function.

    Raises
    ------
    ValueError
        if distance is something else than SPS', 'voice leading', 'tone by tone'
                or 'binary'.

    Returns
    -------
    float
        The corresponding distance between tĥe two chords.

    """
    if triad_reduction:
        chord_type1 = TRIAD_REDUCTION[chord_type1]
        chord_type2 = TRIAD_REDUCTION[chord_type2]

    if distance == "SPS":
        return SPS_distance(
            root1=root1,
            root2=root2,
            chord_type1=chord_type1,
            chord_type2=chord_type2,
            inversion1=inversion1,
            inversion2=inversion2,
            changes1=changes1,
            changes2=changes2,
            **kwargs,
        )

    elif distance == "mechanical":
        return mechanical_distance(
            root1=root1,
            root2=root2,
            chord_type1=chord_type1,
            chord_type2=chord_type2,
            inversion1=inversion1,
            inversion2=inversion2,
            changes1=changes1,
            changes2=changes2,
            **kwargs,
        )

    elif distance == "tone by tone":
        return tone_by_tone_distance(
            root1=root1,
            root2=root2,
            chord_type1=chord_type1,
            chord_type2=chord_type2,
            inversion1=inversion1,
            inversion2=inversion2,
            changes1=changes1,
            changes2=changes2,
            **kwargs,
        )

    elif distance == "binary":
        return (
            0
            if root1 == root2
            and chord_type1 == chord_type2
            and inversion1 == inversion2
            and changes1 == changes2
            else 1
        )

    else:
        raise ValueError(
            "distance must be 'SPS', 'mechanical', 'tone by tone' or 'binary'."
        )
