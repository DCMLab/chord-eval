"""This script can be used to create aligned data tsv files from the raw annotations."""
# The code in this directory is adapted from https://github.com/apmcleod/harmonic-inference v1.0.0

import argparse
from glob import glob
import logging
from pathlib import Path
from tqdm import tqdm

import piece
from eval_utils import get_labels_df
from corpus_reading import load_clean_corpus_dfs, aggregate_annotation_dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create new aligned data tsv files from the raw annotations."
    )

    parser.add_argument(
        "-fh",
        help=(
            "The path to the raw functional-harmony data. The scores should be in "
            "`-fh`/data/BPS/scores/*.mxl"
        ),
        type=Path,
        default=Path("../functional-harmony")
    )

    parser.add_argument(
        "-a",
        "--annotations",
        help="The path to the raw DCML Beethoven repo files.",
        type=Path,
        default=Path("../../beethoven_piano_sonatas"),
    )

    parser.add_argument(
        "-tmp",
        help="The path to a directory used as temporary storage for the Beethoven tsvs.",
        type=Path,
        default=Path("beethoven_corpus_data"),
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The output directory for the generated data files.",
        type=Path,
        default=Path("Beethoven-labels"),
    )

    ARGS = parser.parse_args()

    OUTPUT_DIR = ARGS.output
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Regenerate data from beethoven repo
    print("Aggregating DCML corpus data")
    ANNOTATIONS_PATH = ARGS.annotations
    CORPUS_DIR = ARGS.tmp
    if not CORPUS_DIR.exists():
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_annotation_dfs(ANNOTATIONS_PATH, CORPUS_DIR, notes_only=True)

    print("Reading and cleaning DCML corpus data")
    files_df, measures_df, chords_df, notes_df = load_clean_corpus_dfs(CORPUS_DIR)

    print("Parsing data into Pieces and generating output")
    for fh_filename in tqdm(glob(str(ARGS.fh) + "/data/BPS/scores/*.mxl")):
        music_xml_path = Path(fh_filename)
        label_csv_path = (
            music_xml_path.parent.parent /
            "chords" /
            Path(str(music_xml_path.stem) + ".csv")
        )

        if not label_csv_path.exists():
            logging.error(f"FH Label file {label_csv_path} does not exist. Skipping.")
            continue

        _, number, movement = music_xml_path.stem.split("_")

        dcml_corpus = "beethoven_piano_sonatas"
        dcml_file_name = f"{number}-{movement[-1]}.tsv"

        df = files_df.loc[
            (files_df["corpus_name"] == dcml_corpus) & (files_df["file_name"] == dcml_file_name)
        ]

        if len(df) == 0:
            logging.error(f"No matching DCML file_id found for score {music_xml_path}. Skipping.")
            continue

        fh_score = piece.get_score_piece_from_music_xml(music_xml_path, label_csv_path)

        if len(df) > 1:
            logging.error(
                f"Multiple matching df file_ids found for score {music_xml_path}: {df}."
                "\nUsing the first."
            )

        file_id = df.index[0]

        try:
            dcml_score = piece.get_score_piece_from_data_frames(
                notes_df.loc[file_id],
                chords_df.loc[file_id],
                measures_df.loc[file_id],
                use_suspensions=True,
            )
        except KeyError:
            logging.error(
                f"No matching chord_df data found for score {music_xml_path} (file_id {file_id})."
                " Skipping."
            )
            continue

        fh_label_df = get_labels_df(fh_score, tpc_c=0)
        fh_label_df.to_csv(OUTPUT_DIR / f"fh-{number}-{movement[-1]}.tsv", index=False, sep="\t")

        dcml_label_df = get_labels_df(dcml_score, tpc_c=0)
        dcml_label_df.to_csv(
            OUTPUT_DIR / f"dcml-{number}-{movement[-1]}.tsv", index=False, sep="\t"
        )
