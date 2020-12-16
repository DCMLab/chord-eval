from typing import List

import numpy as np

from data_types import ChordType, PitchType
from constants import NUM_PITCHES, CHORD_PITCHES


def get_chord_pitches(
    root: int,
    chord_type: ChordType,
    inversion: int = 0,
) -> List[int]:
    """
    Get the MIDI note numbers of a given chord root, type, and inversion.

    Parameters
    ----------
    root : int
        The root of the given chord, as MIDI note number. If the chord is some
        inversion, the root pitch will be on this MIDI note, but there may be
        other pitches below it.
    chord_type : ChordType
        The chord type of the given chord.
    inversion : int
        The inversion of the chord.

    Returns
    -------
    chord_pitches : List[int]
        A List of the MIDI note numbers of the notes in the given chord, whose root
        pitch is at the given root. Depending on the inversion, other pitches may be
        below the root.
    """
    # By default with root 0
    chord_pitches = np.array(CHORD_PITCHES[PitchType.MIDI][chord_type], dtype=int)

    # Transpose up to the correct root
    chord_pitches += root

    # Calculate the correct inversion
    for i in range(inversion):
        if i % len(chord_pitches) == 0:
            chord_pitches -= NUM_PITCHES[PitchType.MIDI]
        chord_pitches = np.append(chord_pitches[1:], chord_pitches[0] + NUM_PITCHES[PitchType.MIDI])

    return chord_pitches
