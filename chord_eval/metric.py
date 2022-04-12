#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance

from chord_eval.constants import TRIAD_REDUCTION
from chord_eval.data_types import ChordType, PitchType
from chord_eval.utils import (
    filter_noise,
    find_notes_matching,
    find_peaks,
    get_chord_pitches,
    get_dft_from_chord,
    get_smallest_interval,
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
def voice_leading_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    changes1: str = None,
    changes2: str = None,
    pitch_type: PitchType = PitchType.MIDI,
    only_bass_tpc: bool = True,
    duplicate_bass: bool = True,
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

    pitch_type : PitchType, optional
        The pitch type of the given root. If PitchType.MIDI, the pitches are treated
        as a MIDI note number with C4 = 60. If PitchType.TPC, the pitches are treated
        as an interval above C along the circle of fifths (so G = 1, F = -1, etc.).
        The default is PitchType.MIDI.

    only_bass_tpc : bool, optional
        If PitchType.TPC, this parameter specify if only the bass is treated with
        its TPC representation. The other pitch numbers will correspond to their
        Midi representation. This has been implemented because bass line moves
        often in fifths.

        The default is True.

    duplicate_bass : bool, optional
        If False, the basses are only compared between each other in the first stage :
        for a given pair of chord of same length, it ensures that each notes are
        matched only once.

        however, given a pair of chords with a different number of notes, the
        function will first compare the two basses, then compare the remaining
        sets of notes (without the basses), find the best matching and remove the
        matched notes of the chord with the more notes and finally compare the new
        set of the remaining notes against the full small chord (with its bass)
        until each 'extra' notes are matched.

        Thus, given a pair of chords with a different number of notes, the bass
        of the chord with the more notes will be matched only once but not the
        bass of the small chord.

        If True, the basses are first compare between each other but can be matched
        an other time at each other step : for a given pair of chord of same length,
        both basses can be matched twice ; for a given pair of chords with a different
        number of notes the bass of the small chord can be matched even more times.

        The default is True.

        (The bass_weight only weight the first comparison between the two basses)

    bass_weight : int, optional
        The weight of the basses distance of the two chords. The default is 1.

    Returns
    -------
    total_steps : int
        The number of semitones between the pitches of each chords.

    """
    # note number of the pitches in the chord
    notes1 = pd.Series(
        get_chord_pitches(
            root=root1,
            chord_type=chord_type1,
            pitch_type=pitch_type,
            inversion=inversion1,
            changes=changes1,
        )
    )

    notes2 = pd.Series(
        get_chord_pitches(
            root=root2,
            chord_type=chord_type2,
            pitch_type=pitch_type,
            inversion=inversion2,
            changes=changes2,
        )
    )

    # Bass weighted similarity
    bass_steps = get_smallest_interval(notes1[0], notes2[0])

    # Specify if the bass only is in its TPC representation
    if pitch_type == PitchType.TPC and only_bass_tpc:
        pitch_type = PitchType.MIDI

        notes1 = pd.Series(
            get_chord_pitches(
                root=root1,
                chord_type=chord_type1,
                pitch_type=pitch_type,
                inversion=inversion1,
                changes=changes1,
            )
        )

        notes2 = pd.Series(
            get_chord_pitches(
                root=root2,
                chord_type=chord_type2,
                pitch_type=pitch_type,
                inversion=inversion2,
                changes=changes2,
            )
        )

    # Specify if the bass can be duplicated
    if duplicate_bass:
        total_steps, matching = find_notes_matching(notes1, notes2)

        # if the basses have been matched together, do not count the bass_step
        # an other time
        if (0, 0) in matching:
            total_steps += (bass_weight - 1) * bass_steps
        else:
            total_steps += bass_weight * bass_steps

    else:
        total_steps, matching = find_notes_matching(notes1[1:], notes2[1:])
        total_steps += bass_weight * bass_steps
        matching = [(0, 0)] + [(pair[0] + 1, pair[1] + 1) for pair in matching]

    # Tackle different chord length
    if len(notes1) != len(notes2):

        # Find which chord is the one with the more notes and keep track of its index
        if len(notes1) > len(notes2):
            big_chord = notes1
            bc_idx = 0
            short_chord = notes2
        elif len(notes1) < len(notes2):
            big_chord = notes2
            bc_idx = 1
            short_chord = notes1

        # Remove all the pitches in the biggest chord that have already been
        # matched
        for pair in matching:
            idx = pair[bc_idx]
            big_chord.drop(idx, inplace=True)

        # Find corresponding note in the chord with fewer note for every 'extra'
        # note
        for note_b in big_chord:
            total_steps += min(
                [get_smallest_interval(note_b, note_s) for note_s in short_chord]
            )

    return total_steps


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

    notes1 = (
        get_chord_pitches(
            root=root1,
            chord_type=chord_type1,
            pitch_type=PitchType.MIDI,
            inversion=inversion1,
            changes=changes1,
        )
        % 12
    )

    notes2 = (
        get_chord_pitches(
            root=root2,
            chord_type=chord_type2,
            pitch_type=PitchType.MIDI,
            inversion=inversion2,
            changes=changes2,
        )
        % 12
    )

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

    triad_reduction : bool, optional
        If True, the function will reduce every input chords to their triad
        reduction.
        The default is False.

    **kwargs : TYPE
        Additional argument for the type of metric used.
        If distance is 'SPS', this will be arguments for the SPS_distance function.
        If distance is 'voice leading', this will be arguments for the
        voice_leading_distance function.
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

    elif distance == "voice leading":
        return voice_leading_distance(
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
            "distance must be 'SPS', 'voice leading', 'tone by tone' or 'binary'."
        )
