"""Utility functions for evaluating model outputs."""
from typing import Tuple

import harmonic_constants as hc
import harmonic_utils as hu
import numpy as np
import pandas as pd
from chord import Chord
from data_types import PitchType
from piece import Piece


def get_labels_df(piece: Piece, tpc_c: int = hc.TPC_C) -> pd.DataFrame:
    """
    Create and return a labels_df for a given Piece, containing all chord and key
    information for each segment of the piece, in all formats (TPC and MIDI pitch).

    Parameters
    ----------
    piece : Piece
        The piece to create a labels_df for.
    tpc_c : int
        Where C should be in the TPC output.

    Returns
    -------
    labels_df : pd.DataFrame
        A labels_df, with the columns:
            - chord_root_tpc
            - chord_root_midi
            - chord_type
            - chord_inversion
            - chord_suspension_midi
            - chord_suspension_tpc
            - key_tonic_tpc
            - key_tonic_midi
            - key_mode
            - duration
            - mc
            - onset_mc
    """

    def get_suspension_strings(chord: Chord) -> Tuple[str, str]:
        """
        Get the tpc and midi strings for the given chord's suspension and changes.

        Parameters
        ----------
        chord : Chord
            The chord whose string to return.

        Returns
        -------
        tpc_string : str
            A string representing the mapping of altered pitches in the given chord.
            Each altered pitch is represented as "orig:new", where orig is the pitch in the default
            chord voicing, and "new" is the altered pitch that is actually present. For added
            pitches, "orig" is the empty string. "new" can be prefixed with a "+", in which
            case this pitch is present in an upper octave. Pitches are represented as TPC,
            and multiple alterations are separated by semicolons.
        midi_string : str
            The same format as tpc_string, but using a MIDI pitch representation.
        """
        if chord.suspension is None:
            return "", ""

        change_mapping = hu.get_added_and_removed_pitches(
            chord.root,
            chord.chord_type,
            chord.suspension,
            chord.key_tonic,
            chord.key_mode,
        )

        mappings_midi = []
        mappings_tpc = []
        for orig, new in change_mapping.items():
            if orig == "":
                orig_midi = ""
                orig_tpc = ""
            else:
                orig_midi = str(
                    hu.get_pitch_from_string(
                        hu.get_pitch_string(int(orig), PitchType.TPC), PitchType.MIDI
                    )
                )
                orig_tpc = str(int(orig) - hc.TPC_C + tpc_c)

            prefix = ""
            if new[0] == "+":
                prefix = "+"
                new = new[1:]

            new_midi = prefix + str(
                hu.get_pitch_from_string(
                    hu.get_pitch_string(int(new), PitchType.TPC), PitchType.MIDI
                )
            )
            new_tpc = prefix + str(int(new) - hc.TPC_C + tpc_c)

            mappings_midi.append(f"{orig_midi}:{new_midi}")
            mappings_tpc.append(f"{orig_tpc}:{new_tpc}")

        return ";".join(mappings_tpc), ";".join(mappings_midi)

    labels_list = []

    chords = piece.get_chords()
    onsets = [note.onset for note in piece.get_inputs()]
    chord_changes = piece.get_chord_change_indices()
    chord_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    chord_suspensions_midi = np.full(len(piece.get_inputs()), "", dtype=object)
    chord_suspensions_tpc = np.full(len(piece.get_inputs()), "", dtype=object)
    chord_label_str = np.full(len(piece.get_inputs()), "", dtype=object)
    for chord, start, end in zip(chords, chord_changes, chord_changes[1:]):
        chord_labels[start:end] = chord.get_one_hot_index(
            relative=False, use_inversion=True, pad=False
        )

        tpc_string, midi_string = get_suspension_strings(chord)

        chord_suspensions_tpc[start:end] = tpc_string
        chord_suspensions_midi[start:end] = midi_string

        chord_label_str[start:end] = chord.label

    chord_labels[chord_changes[-1] :] = chords[-1].get_one_hot_index(
        relative=False, use_inversion=True, pad=False
    )

    tpc_string, midi_string = get_suspension_strings(chords[-1])

    chord_suspensions_tpc[chord_changes[-1] :] = tpc_string
    chord_suspensions_midi[chord_changes[-1] :] = midi_string
    chord_label_str[chord_changes[-1] :] = chords[-1].label

    keys = piece.get_keys()
    key_changes = piece.get_key_change_input_indices()
    key_labels = np.zeros(len(piece.get_inputs()), dtype=int)
    for key, start, end in zip(keys, key_changes, key_changes[1:]):
        key_labels[start:end] = key.get_one_hot_index()
    key_labels[key_changes[-1] :] = keys[-1].get_one_hot_index()

    chord_labels_list = hu.get_chord_from_one_hot_index(
        slice(len(hu.get_chord_label_list(PitchType.TPC))), PitchType.TPC
    )
    key_labels_list = hu.get_key_from_one_hot_index(
        slice(len(hu.get_key_label_list(PitchType.TPC))), PitchType.TPC
    )

    for (
        duration,
        chord_label,
        key_label,
        suspension_tpc,
        suspension_midi,
        onset,
        label_str,
    ) in zip(
        piece.get_duration_cache(),
        chord_labels,
        key_labels,
        chord_suspensions_tpc,
        chord_suspensions_midi,
        onsets,
        chord_label_str,
    ):
        if duration == 0:
            continue

        root_tpc, chord_type, inversion = chord_labels_list[chord_label]
        tonic_tpc, mode = key_labels_list[key_label]

        root_midi = hu.get_pitch_from_string(
            hu.get_pitch_string(root_tpc, PitchType.TPC), PitchType.MIDI
        )
        tonic_midi = hu.get_pitch_from_string(
            hu.get_pitch_string(tonic_tpc, PitchType.TPC), PitchType.MIDI
        )

        labels_list.append(
            {
                "chord_root_tpc": root_tpc - hc.TPC_C + tpc_c,
                "chord_root_midi": root_midi,
                "chord_type": chord_type,
                "chord_inversion": inversion,
                "chord_suspension_tpc": suspension_tpc,
                "chord_suspension_midi": suspension_midi,
                "key_tonic_tpc": tonic_tpc - hc.TPC_C + tpc_c,
                "key_tonic_midi": tonic_midi,
                "key_mode": mode,
                "duration": duration,
                "mc": onset[0],
                "mn_onset": onset[1],
                "label": label_str,
            }
        )

    return pd.DataFrame(labels_list)
