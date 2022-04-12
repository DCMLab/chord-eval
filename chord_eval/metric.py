#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from functools import lru_cache
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pretty_midi
from librosa import cqt, stft, vqt
from librosa.feature import melspectrogram
from scipy.signal import medfilt
from scipy.spatial import distance

from chord_eval.constants import TRIAD_REDUCTION
from chord_eval.data_types import ChordType, PitchType
from chord_eval.utils import get_chord_pitches


def create_chord(
    root: int,
    chord_type: ChordType,
    inversion: int = 0,
    program: int = 0,
    changes: str = None,
    pitches: Tuple[int] = None,
) -> pretty_midi.PrettyMIDI:
    """
    Create a pretty_midi object with an instrument and the given chord for 1 second.

    Parameters
    ----------
    root : int
        The root of the given chord, as MIDI note number. If the chord
        is some inversion, the root pitch will be on this MIDI note, but there
        may be other pitches below it.

    chord_type : ChordType
        The chord type of the given chord.

    inversion : int, optional
        The inversion of the chord. The default is 0.

    changes : str
        Any alterations to the chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    program : int, optional
        The general MIDI program number used by the fluidsynth to synthesize
        the wave form of the MIDI data of the first chord (it uses the
        TimGM6mb.sf2 sound font file included with pretty_midi and synthesizes
        at fs=44100hz by default)

        The general MIDI program number (instrument index) is in [0, 127] :
        https://pjb.com.au/muscript/gm.html

        The default is 0 for piano.

    pitches : Tuple[int]
        A Tuple of possible absolute MIDI pitch numbers to use in this chord.
        If given, only a subset of these pitches will be included in the returned
        chord. Specifically, those which share a pitch class with any of the
        default chord tones. Note that this means that some default chord tones might
        not be present in the returned chord.

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
        A pretty_midi object containing the instrument and the chord to play.

    """
    # MIDI object
    pm = pretty_midi.PrettyMIDI()

    # Instrument instance
    instrument = pretty_midi.Instrument(program=program)

    # note number of the pitches in the chord
    notes = get_chord_pitches(
        root=root,
        chord_type=chord_type,
        pitch_type=PitchType.MIDI,
        inversion=inversion,
        changes=changes,
    )
    notes += 60  # centered around C4

    if pitches is not None:
        notes = [pitch for pitch in pitches if np.any(notes % 12 == pitch % 12)]

    for note_number in notes:
        # Note instance of 1 second
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=0, end=1)
        instrument.notes.append(note)

    pm.instruments.append(instrument)

    return pm


def get_dft_from_MIDI(
    midi_object: pretty_midi.PrettyMIDI,
    transform: str = "vqt",
    hop_length: int = 512,
    bins_per_octave: int = 60,
    n_mels: int = 512,
) -> np.ndarray:
    """
    Get the discrete Fourier transform of the synthesized instrument's notes
    contained in the MIDI object using fluidsynth.

    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        MIDI object that should contain at least one instrument which should
        contain a program and at least one note.
    transform : str
        Indicates which transform to use to get the spectrogram of the
        synthesized instrument's notes. It can either be 'stft, 'cqt', 'vqt'
        or 'mel'/'melspectrogram'.
        The default is 'vqt'
    hop_length : int
        Number of samples between successive CQT or VQT columns if transform is
        'cqt' or 'vqt'.
        the default is 512.
    bin_per_octave : int
        Number of bins per octave if transform is 'cqt' or 'vqt'.
        the default is 60.
    n_mels : int
        Number of Mel bands to generate if transform is 'mel' or 'melspectrogram'.
        the default is 512.

    Returns
    -------
    dft : np.ndarray
        The discrete Fourier transform.

    """
    sr = 22050  # Librosa's default sampling rate

    y = midi_object.fluidsynth(
        fs=sr
    )  # The sampling rate should match Librosa's default

    # fuidsynth synthesizes the 1s-duration note and its release for an additional second.
    # Only keep the first second of the signal.
    y = y[:sr]

    if transform == "stft":
        dft = np.abs(stft(y))
    elif transform == "cqt":
        dft = np.abs(
            cqt(
                y,
                hop_length=hop_length,
                n_bins=bins_per_octave * 7,
                bins_per_octave=bins_per_octave,
            )
        )
    elif transform == "vqt":
        dft = np.abs(
            vqt(
                y,
                hop_length=hop_length,
                n_bins=bins_per_octave * 7,
                bins_per_octave=bins_per_octave,
            )
        )
    elif transform in ["mel", "melspectrogram"]:
        dft = np.abs(melspectrogram(y, n_mels=n_mels))
    else:
        raise ValueError(
            "Transform must be 'stft', 'cqt', 'vqt', 'mel' or 'melspectrogram'."
        )

    # Return only the middle frame of the spectrogram
    middle_idx = int(np.floor(dft.shape[1] / 2))
    dft_middle = dft[:, middle_idx]

    return dft_middle


@lru_cache
def get_dft_from_chord(
    root: int,
    chord_type: ChordType,
    inversion: int = 0,
    changes: str = None,
    pitches: Tuple[int] = None,
    program: int = 0,
    transform: str = "vqt",
    hop_length: int = 512,
    bins_per_octave: int = 60,
    n_mels: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the discrete Fourier transform of the given chord and its discrete Fourier
    transform an octave below and above (since the spectrum content is different
    from one octave to the other), by creating a pretty_MIDI object and calling
    the get_dft_from_MIDI function.

    Parameters
    ----------
    root : int
        The root of the given chord, as MIDI note number. If the chord
        is some inversion, the root pitch will be on this MIDI note, but there
        may be other pitches below it.

    chord_type : ChordType
        The chord type of the given chord.

    inversion : int, optional
        The inversion of the chord.
        The default is 0.

    changes : str
        Any alterations to the chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    pitches : Tuple[int]
        A Tuple of possible absolute MIDI pitch numbers to use in this chord.
        If given, only a subset of these pitches will be included in the generated
        chord. Specifically, those which share a pitch class with any of the
        default chord tones. Note that this means that some default chord tones might
        not be present in the generated chord.

    program : int, optional
        The general MIDI program number used by the fluidsynth to synthesize
        the wave form of the MIDI data of the given chord (it uses the
        TimGM6mb.sf2 sound font file included with pretty_midi and synthesizes
        at fs=44100hz by default)
        The general MIDI program number (instrument index) is in [0, 127] :
        https://pjb.com.au/muscript/gm.html
        The default is 0.

    transform : str
        Indicates which transform to use to get the spectrogram of the
        synthesized instrument's notes. It can either be 'stft, 'cqt', 'vqt'
        or 'mel'/'melspectrogram'.
        The default is 'vqt'

    hop_length : int
        Number of samples between successive CQT or VQT columns if transform is
        'cqt' or 'vqt'.
        the default is 512.

    bin_per_octave : int
        Number of bins per octave if transform is 'cqt' or 'vqt'.
        the default is 60.

    n_mels : int
        Number of Mel bands to generate if transform is 'mel' or 'melspectrogram'.
        the default is 512.

    Returns
    -------
    dft_below : np.ndarray
            the discrete Fourier transform of the chord an octave below.
    dft : np.ndarray
            the discrete Fourier transform of the chord.
    dft_above : np.ndarray
            the discrete Fourier transform of the chord an octave above.
    """
    # MIDI object of the chord
    pms = [
        create_chord(
            root=r,
            chord_type=chord_type,
            inversion=inversion,
            changes=changes,
            program=program,
            pitches=pitches,
        )
        for r in [root - 12, root, root + 12]
    ]

    # Spectrum of the synthesized instrument's notes of the MIDI object
    dfts = [
        get_dft_from_MIDI(
            pm,
            transform=transform,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_mels=n_mels,
        )
        for pm in pms
    ]

    return dfts


def filter_noise(
    dft: np.ndarray,
    size_med: int = 41,
    noise_factor: float = 4.0,
) -> np.ndarray:
    """
    Filter the discrete Fourier transform passed in argument : For each
    frequency bin of the spectrum, the median magnitude is calculated across a
    centered window (the Fourier transform is zero-padded). If the magnitude
    of that component is less that a noise-factor times the window’s median,
    it is considered noise and removed (setting their vaue to 0).

        function inspired by the noiseSignal function from the Music-Perception-Toolbox
        by Andrew J. Milne, The MARCS Institute, Western Sydney University :
        https://github.com/andymilne/Music-Perception-Toolbox/blob/master/noiseSignal.m


    Parameters
    ----------
    dft : np.ndarray
        The discrete Fourier transform.
    size_med : int
        The width of the centered window over which the median is calculated
        for each component of the spectrum. The default is 41 bins.
    noise_factor : float
        The noise threshold. The default is 4.0.

    Returns
    -------
    dft_filtered : np.ndarray
        The filtered discrete Fourier transform.

    """
    noise_floor = medfilt(dft, size_med)
    dft_filtered = np.zeros_like(dft)

    unfiltered_mask = dft >= noise_factor * noise_floor
    dft_filtered[unfiltered_mask] = dft[unfiltered_mask]

    return dft_filtered


def find_peaks(dft: np.ndarray) -> np.ndarray:
    """
    Isolate the peaks of a spectrum : If there are multiple non-zero consecutive
    frequency bins in the spectrum, the function keeps only the maximum of all
    these bins and filter out all the others by setting their value to 0.

    This function should be use after having used the filter_noise function.

    function inspired by the noiseSignal function from the Music-Perception-Toolbox
    by Andrew J. Milne, The MARCS Institute, Western Sydney University :
    https://github.com/andymilne/Music-Perception-Toolbox/blob/master/noiseSignal.m

    Parameters
    ----------
    dft : np.ndarray
        The discrete Fourier transform.

    Returns
    -------
    peaks : np.ndarray
        The filtered discrete Fourier transform.
    """
    # 1 at the start of a non-zero section, -1 at the end of one
    dft_binary_diff = np.diff(np.array(np.pad(dft, 1) > 0, dtype=int))

    peaks_start_idx = np.where(dft_binary_diff == 1)[0]
    peaks_end_idx = np.where(dft_binary_diff == -1)[0]

    peak_idx = np.array(
        [
            start + np.argmax(dft[start:end])
            for start, end in zip(peaks_start_idx, peaks_end_idx)
        ]
    )

    peaked_dft = np.zeros_like(dft)
    peaked_dft[peak_idx] = dft[peak_idx]

    return peaked_dft


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


def get_smallest_interval(note1: int, note2: int) -> int:
    """
    Find the smallest interval between tow given midi notes by shifting them by
    one or more octaves.

    Parameters
    ----------
    note1 : int
        Midi number of the first note.
    note2 : int
        Midi number of the second note

    Returns
    -------
    int
        Midi number corresponding to the smallest interval.

    """
    diff = np.abs(note1 - note2) % 12
    return min(diff, 12 - diff)


def find_notes_matching(notes1: np.ndarray, notes2: np.ndarray):
    """
    Find the smallest overall number of semi-tones between each lists of notes
    passed in argument and the corresponding matching.

    Parameters
    ----------
    notes1 : np.ndarray
            List of notes.
    notes2 : np.ndarray
            List of notes.

    Returns
    -------
    total_steps : int
            smallest overall number of semi-tones between each list of notes.
    matching : List
            corresponding matching.

    """
    # Matrix that keep the number of semi-tones between each pair of pitches between
    # the two chords.
    similarities = np.ndarray((len(notes1), len(notes2)))

    # Graph for the matching
    G = nx.Graph()

    for idx1, note1 in enumerate(notes1):
        for idx2, note2 in enumerate(notes2):

            similarity = 6 - get_smallest_interval(
                note1, note2
            )  # 6 being the maximum number
            # semi-tones between two pitches
            # convertion of a distance to similarity

            similarities[idx1, idx2] = similarity

            # the second node is define by the length of the first chord plus
            # the index of the corresponding pitch within the second chord
            G.add_weighted_edges_from([(idx1, idx2 + len(notes1), similarity)])

    matching = nx.max_weight_matching(G, maxcardinality=True)
    # We substract the length of the first chord to the second node of the
    # matching to the index of the corresponding pitch within the second chord
    matching = [
        (pair[0], pair[1] - len(notes1))
        if pair[0] < pair[1]
        else (pair[1], pair[0] - len(notes1))
        for pair in matching
    ]

    # Total distance
    total_steps = 0
    for pair in matching:
        total_steps += 6 - similarities[pair[0], pair[1]]

    return total_steps, matching


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
    root_weight: int = 1,
    bass_weight: int = 1,
) -> float:
    """
    Get the tone by tone distance between two chords : the number
    of matched pitches between the chords over the maximum number of matched pitch.


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

    root_weight : int, optional
        The weight of the matching of the two chords' root.
        The default is 1.

    base_weight : int, optional
        The weight of the matching of the two chords' bass (the lowest note of
                chord).
        The default is 1.

    Returns
    -------
    dist : float
        The tone by tone distance between two chords.

    """
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

    matches = sum([1 if note in notes2 else 0 for note in notes1])

    if root1 == root2:
        matches += root_weight - 1  # don't count the root match one more time
    if notes1[0] == notes2[0]:
        matches += bass_weight - 1  # don't count the bass match one more time

    dist = 1 - matches / (max(len(notes1), len(notes2)) + root_weight + bass_weight - 2)

    return dist


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
