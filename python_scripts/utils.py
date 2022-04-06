import logging
from typing import List

import numpy as np
from constants import CHORD_PITCHES, NUM_PITCHES, TPC_C
from data_types import ChordType, PitchType


def get_chord_pitches(
    root: int,
    chord_type: ChordType,
    pitch_type: PitchType,
    inversion: int = 0,
    changes: str = None,
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
    chord_pitches : List[int]
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
