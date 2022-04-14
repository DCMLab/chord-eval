import logging
from functools import lru_cache
from typing import List, Tuple

import networkx as nx
import numpy as np
import pretty_midi
from librosa import cqt, stft, vqt
from librosa.feature import melspectrogram
from scipy.signal import medfilt

from chord_eval.constants import CHORD_PITCHES, NUM_PITCHES, TPC_C
from chord_eval.data_types import ChordType, PitchType


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
        for r in ([root - 12, root, root + 12] if pitches is None else [root])
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


@lru_cache
def get_chord_pitches(
    root: int,
    chord_type: ChordType,
    pitch_type: PitchType,
    inversion: int = 0,
    changes: str = None,
) -> np.ndarray:
    """
    Get the MIDI note numbers of a given chord root, type, and inversion.

    Parameters
    ----------
    root : int
        The root of the given chord, either as a MIDI note number or as an interval
        above C along the circle of fifths (see the `pitch_type` parameter).

    chord_type : ChordType
        The chord type of the given chord.

    pitch_type : PitchType
        The pitch type of the given root. If PitchType.MIDI, the root is treated
        as a MIDI note number with C4 = 60. If PitchType.TPC, the root is treated
        as an interval above C along the circle of fifths (so G = 1, F = -1, etc.)

    inversion : int
        The inversion of the chord.

    changes : str
        Any alterations to the chord's pitches, as a semi-colon separated string.
        Each alteration should be in the form "orig:new", where "orig" represents
        the original pitch that has been altered to "new". "orig" can also be blank
        for added pitches, and "new" can be prepended with "+" to indicate that a
        pitch occurs in an upper octave (e.g., a C7 chord with a 9th is represented
        by ":+1", using MIDI pitch). Note that TPC pitch does not allow for the
        representation of different octaves so any "+" is ignored.

    Returns
    -------
    chord_pitches : np.ndarray
        A List of pitches of the notes in the given chord, whose root
        pitch is at the given root. The pitches will either be MIDI note numbers
        (if pitch_type == PitchType.MIDI) or intervals above C along the circle of
        fifths (if pitch_type == PitchType.TPC). Depending on the inversion, other
        pitches may be below the root.
    """
    # By default with root C (0 for MIDI, TPC_C for TPC)
    chord_pitches = np.array(CHORD_PITCHES[pitch_type][chord_type], dtype=int)

    if pitch_type == PitchType.TPC:
        # For TPC, shift so that C is at 0
        chord_pitches -= TPC_C

    # Transpose up to the correct root
    chord_pitches += root

    # Calculate the correct inversion
    if pitch_type == PitchType.MIDI:
        # For MIDI note number, notes shift up by octaves for inversions
        for i in range(inversion):
            if i % len(chord_pitches) == 0:
                chord_pitches -= NUM_PITCHES[PitchType.MIDI]
            chord_pitches = np.append(
                chord_pitches[1:],
                chord_pitches[0] + NUM_PITCHES[PitchType.MIDI],
            )

    else:
        # For TPC, we just have to rotate the list of pitches
        chord_pitches = np.roll(chord_pitches, -inversion)

    # Handle changes
    if changes is not None:
        if pitch_type == PitchType.TPC:
            # TPC doesn't support higher octaves
            if "+" in changes:
                logging.warning(
                    "TPC pitch doesn't support upper octave changes with +."
                    "Ignoring the +."
                )
                changes = changes.replace("+", "")

        for change in changes.split(";"):
            orig, new = change.split(":")

            if pitch_type == PitchType.MIDI:
                # Handle octave up and convert new to int
                if new[0] == "+":
                    octave_up = 12
                    new = int(new[1:])
                else:
                    octave_up = 0
                    new = int(new)

                bass_note = chord_pitches[0]

                # This finds the pitch in the octave immediately above the bass note
                # This is correct unless the bass note is replaced by a lower neighbor
                # The bass-note lower neighbor case is handled below during replacement
                interval_above_bass = (new % 12) - (bass_note % 12) % 12

                new = bass_note + interval_above_bass + octave_up

            else:
                # No need to worry about octaves for TPC pitch
                new = int(new)

            if orig == "":
                # Simply add the new pitch
                chord_pitches = np.append(chord_pitches, new)

            else:
                # The new pitch is a replacement
                if pitch_type == PitchType.TPC:
                    try:
                        index = np.where(chord_pitches == int(orig))[0][0]
                    except IndexError:
                        logging.error(
                            f"Replaced pitch {orig} not in chord pitches: {chord_pitches}."
                            f"Skipping change {change}."
                        )
                        continue

                else:
                    try:
                        index = np.where(chord_pitches % 12 == int(orig) % 12)[0][0]
                    except IndexError:
                        logging.error(
                            f"Replaced pitch {orig} not in chord pitches: {chord_pitches}."
                            f"Skipping change {change}."
                        )
                        continue

                    if index == 0 and new - chord_pitches[0] > 6:
                        # We need to handle the special case of a bass-note lower neighbor
                        new -= 12

                chord_pitches[index] = new

    return chord_pitches


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


def find_notes_matching(
    notes1: np.ndarray, notes2: np.ndarray, distance: "Distance"
) -> Tuple[int, List[int]]:
    """
    Find the smallest distance between two lists of notes.

    Parameters
    ----------
    notes1 : np.ndarray
        List of notes.
    notes2 : np.ndarray
        List of notes.
    distance : Distance
        The distance metric to use.

    Returns
    -------
    total_steps : int
        The minimum length between the two sets of notes according to the distance metric.
    matching : List[Tuple[int, int]]
        A List of matched pairs of ids from notes1 and notes2.

    """
    # Matrix that keep the similarity between each pair of notes
    similarities = np.zeros((len(notes1), len(notes2)))

    for idx1, pitch1 in enumerate(notes1):
        for idx2, pitch2 in enumerate(notes2):
            similarity = -distance.distance_between(pitch1, pitch2)
            similarities[idx1, idx2] = similarity

    # Graph for the matching
    G = nx.Graph()
    # The second node ID is the length of the first chord plus its index
    G.add_weighted_edges_from(
        [
            (idx1, idx2 + len(notes1), similarities[idx1, idx2])
            for idx1 in range(len(notes1))
            for idx2 in range(len(notes2))
        ]
    )

    matching = nx.max_weight_matching(G, maxcardinality=True)

    # Subtract the length of the first chord from the 2nd node IDs
    match_ids = [
        (
            (pair[0], pair[1] - len(notes1))
            if pair[0] < pair[1]
            else (pair[1], pair[0] - len(notes1))
        )
        for pair in matching
    ]

    # Total distance
    total_steps = -sum([similarities[pair[0], pair[1]] for pair in match_ids])

    return total_steps, match_ids


class Distance:
    """
    Distance objects define a spatial distance measure in semitone space.
    """

    def __init__(self, adjacent_step: int = 1, custom_distance: List[int] = None):
        """
        Create a new Distance object, either by defining which step should be adjacent,
        or by defining a custom distance List manually.

        Parameters
        ----------
        adjacent_step : int
            How many semitones apart should be adjacent according to this distance.
            For example, to measure distance in (enharmonic) fifths, set this to 7.

            No factor of 12 (greater than 1) will produce a valid, usable Distance
            measure, since they will not assign a difference to every semitone.
            For example, 4 (a minor 3rd) is not valid, since you cannot reach ever
            enharmonic interval via minor thirds (C to D, for example).

            However, the object can still be created, and then combined with another
            Distance object via the combine function.

        custom_distance : List[int]
            Manually define a distance list. This should be a list of length 12,
            and if given supercedes adjacency_step. Invalid distances (which must be
            combined with another Distance object) should be assigned -1. The ith element
            represents the distance to the pitch class up i semitones.
        """
        if custom_distance is None:
            distance_up = [0] + [-1] * 11
            idx = adjacent_step % 12
            steps = 1
            while distance_up[idx] == -1:
                distance_up[idx] = steps
                steps += 1
                idx = (idx + adjacent_step) % 12

            distance_down = [0] + [-1] * 11
            idx = -adjacent_step % 12
            steps = 1
            while distance_down[idx] == -1:
                distance_down[idx] = steps
                steps += 1
                idx = (idx - adjacent_step) % 12

            self.distance = (
                Distance(custom_distance=distance_up)
                .combine(Distance(custom_distance=distance_down))
                .distance
            )

        else:
            assert len(custom_distance) == 12, "custom_distance must be of length 12."
            self.distance = custom_distance

    def is_valid(self) -> bool:
        """
        Check whether this Distance object is valid for measuring distances. A valid Distance
        object must have all distances >= 0.

        Returns
        -------
        is_valid : bool
            True if this Distance object is valid. False otherwise.
        """
        return min(self.distance) >= 0

    def combine(self, other_distance: "Distance") -> "Distance":
        """
        Combine this Distance object with another, returning a 3rd whose distances are the
        minimums of the distances from each Distance object at each step.

        Parameters
        ----------
        other_distance : Distance
            The Distance object to combine with.

        Returns
        -------
        distance : Distance
            A new Distance object, whose distance at each step is the minimum of self.distance
            and other_distance.distance.
        """
        return Distance(
            custom_distance=[
                min(d1, d2) if -1 not in [d1, d2] else max(d1, d2)
                for d1, d2 in zip(self.distance, other_distance.distance)
            ]
        )

    def distance_between(self, pitch1: int, pitch2: int) -> int:
        """
        Get the distance between two pitches according to this Distance object.

        Parameters
        ----------
        pitch1 : int
            The first pitch.
        pitch2 : int
            The second pitch.

        Returns
        -------
        distance : int
            The distance between pitch1 and pitch2.
        """
        return self.distance[get_smallest_interval(pitch1, pitch2)]

    def __hash__(self) -> int:
        return tuple(self.distance).__hash__()
