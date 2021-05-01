from typing import List

import numpy as np

from data_types import ChordType, PitchType
from constants import NUM_PITCHES, CHORD_PITCHES, TPC_C


def get_chord_pitches(
    root: int,
    chord_type: ChordType,
    pitch_type: PitchType,
    inversion: int = 0,
) -> List[int]:
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

    Returns
    -------
    chord_pitches : List[int]
        A List of pitches of the notes in the given chord, whose root
        pitch is at the given root. The pitches will either be MIDI note numbers
        (if pitch_type == PitchType.MIDI) or intervals above C along the circle of
        fifths (if pitch_type == PitchType.TPC). Depending on the inversion, other
        pitches may be below the root.
    """
    # By default with root 0
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

    return chord_pitches
