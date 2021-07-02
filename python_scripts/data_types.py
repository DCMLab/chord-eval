from enum import Enum


class PitchType(Enum):
    """
    An Enum representing the way pitches (or tonics, or chord roots) are represented.
    Either TPC (tonal pitch class) or MIDI.
    """

    TPC = 0
    MIDI = 1


class ChordType(Enum):
    """
    The types of chords that are supported.
    """

    MAJOR = 0
    MINOR = 1
    DIMINISHED = 2
    AUGMENTED = 3
    MAJ_MAJ7 = 4
    MAJ_MIN7 = 5
    MIN_MAJ7 = 6
    MIN_MIN7 = 7
    DIM7 = 8
    HALF_DIM7 = 9
    AUG_MIN7 = 10
    AUG_MAJ7 = 11
