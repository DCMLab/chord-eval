from data_types import ChordType, PitchType

TPC_NATURAL_PITCHES = 7
TPC_ACCIDENTALS = 5  # bb, b, natural, #, ##. natural must be in the exact middle
TPC_C_WITHIN_PITCHES = 1
TPC_C = int(TPC_ACCIDENTALS / 2) * TPC_NATURAL_PITCHES + TPC_C_WITHIN_PITCHES


STRING_TO_PITCH = {
    PitchType.TPC: {
        "A": TPC_C + 3,
        "B": TPC_C + 5,
        "C": TPC_C,
        "D": TPC_C + 2,
        "E": TPC_C + 4,
        "F": TPC_C - 1,
        "G": TPC_C + 1,
    },
    PitchType.MIDI: {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11},
}

NUM_PITCHES = {PitchType.TPC: TPC_NATURAL_PITCHES * TPC_ACCIDENTALS, PitchType.MIDI: 12}

ACCIDENTAL_ADJUSTMENT = {PitchType.TPC: TPC_NATURAL_PITCHES, PitchType.MIDI: 1}

# Triad types indexes of ones for a C chord of the given type in a one-hot presence vector
CHORD_PITCHES = {}
for pitch_type in [PitchType.MIDI, PitchType.TPC]:
    CHORD_PITCHES[pitch_type] = {
        ChordType.MAJOR: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"],
            STRING_TO_PITCH[pitch_type]["G"],
        ],
        ChordType.MINOR: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"] - ACCIDENTAL_ADJUSTMENT[pitch_type],
            STRING_TO_PITCH[pitch_type]["G"],
        ],
        ChordType.DIMINISHED: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"] - ACCIDENTAL_ADJUSTMENT[pitch_type],
            STRING_TO_PITCH[pitch_type]["G"] - ACCIDENTAL_ADJUSTMENT[pitch_type],
        ],
        ChordType.AUGMENTED: [
            STRING_TO_PITCH[pitch_type]["C"],
            STRING_TO_PITCH[pitch_type]["E"],
            STRING_TO_PITCH[pitch_type]["G"] + ACCIDENTAL_ADJUSTMENT[pitch_type],
        ],
    }

    # Add major triad 7th chords
    for chord in [ChordType.MAJ_MAJ7, ChordType.MAJ_MIN7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][
            ChordType.MAJOR
        ].copy()

    # Add minor triad 7th chords
    for chord in [ChordType.MIN_MAJ7, ChordType.MIN_MIN7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][
            ChordType.MINOR
        ].copy()

    # Add diminished triad 7th chords
    for chord in [ChordType.DIM7, ChordType.HALF_DIM7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][
            ChordType.DIMINISHED
        ].copy()

    # Add augmented triad 7th chords
    for chord in [ChordType.AUG_MAJ7, ChordType.AUG_MIN7]:
        CHORD_PITCHES[pitch_type][chord] = CHORD_PITCHES[pitch_type][
            ChordType.AUGMENTED
        ].copy()

    # Add major 7ths
    for chord in [ChordType.MAJ_MAJ7, ChordType.MIN_MAJ7, ChordType.AUG_MAJ7]:
        CHORD_PITCHES[pitch_type][chord].append(STRING_TO_PITCH[pitch_type]["B"])

    # Add minor 7ths
    for chord in [
        ChordType.MAJ_MIN7,
        ChordType.MIN_MIN7,
        ChordType.HALF_DIM7,
        ChordType.AUG_MIN7,
    ]:
        CHORD_PITCHES[pitch_type][chord].append(
            STRING_TO_PITCH[pitch_type]["B"] - ACCIDENTAL_ADJUSTMENT[pitch_type]
        )

    # Add diminished 7ths
    for chord in [ChordType.DIM7]:
        CHORD_PITCHES[pitch_type][chord].append(
            STRING_TO_PITCH[pitch_type]["B"] - 2 * ACCIDENTAL_ADJUSTMENT[pitch_type]
        )


NO_REDUCTION = {chord_type: chord_type for chord_type in ChordType}


TRIAD_REDUCTION = {
    ChordType.MAJOR: ChordType.MAJOR,
    ChordType.MINOR: ChordType.MINOR,
    ChordType.DIMINISHED: ChordType.DIMINISHED,
    ChordType.AUGMENTED: ChordType.AUGMENTED,
    ChordType.MAJ_MAJ7: ChordType.MAJOR,
    ChordType.MAJ_MIN7: ChordType.MAJOR,
    ChordType.MIN_MAJ7: ChordType.MINOR,
    ChordType.MIN_MIN7: ChordType.MINOR,
    ChordType.DIM7: ChordType.DIMINISHED,
    ChordType.HALF_DIM7: ChordType.DIMINISHED,
    ChordType.AUG_MIN7: ChordType.AUGMENTED,
    ChordType.AUG_MAJ7: ChordType.AUGMENTED,
}
