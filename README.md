# The chord_eval toolkit

This repository contains the code to evaluate the distances between chords in 4 ways, which are detailed in our paper:

McLeod, Andrew, Suermondt, Xavier, Rammos, Yannis, Herff, Steffen, and Rohrmeier, Martin A. 2022. Three Metrics for Musical
Chord Label Evaluation. In Forum for Information Retrieval Evaluation (FIRE ’22), December 9–13, 2022, Kolkata, India. ACM, New York,
NY, USA, 11 pages. https://doi.org/10.1145/3574318.3574335

## Usage

All of the metrics are implemented in the [`metric.py`](https://github.com/DCMLab/chord-eval/blob/main/chord_eval/metric.py) file. The method [`get_distance`](https://github.com/DCMLab/chord-eval/blob/main/chord_eval/metric.py#L526) can be used to get the distance between two chords.

The `distance` (str) parameter allows you to choose one of our defined distance metrics (see the paper for details):
- `binary`: The binary distance between 2 chords. Returns 0 if the chords are identical and 1 otherwise.
- `SPS`
- `tone by tone`
- `mechanical`

For details on the various parameters, see the `get_distance` documentation, as well as the documentation for each of the individual distance methods: [`SPS_distance`](https://github.com/DCMLab/chord-eval/blob/main/chord_eval/metric.py#L24), [`mechanical_distance`](https://github.com/DCMLab/chord-eval/blob/main/chord_eval/metric.py#L186), and [`tone_by_tone_distance`](https://github.com/DCMLab/chord-eval/blob/main/chord_eval/metric.py#L325).

## Installation

### Download [FluidSynth](https://www.fluidsynth.org)

FluidSynth is needed to get the SPS distance between 2 chords. It allows to synthesize chords to get their spectrum. To download it, run the following command if you using Ubuntu or Debian:

```
sudo apt-get install fluidsynth
```

For other OS, click [here](https://github.com/FluidSynth/fluidsynth/wiki/Download) to see the downloading procedure.

### Conda environment and python packages:

Here are the steps to build a new conda environment with the needed python packages:
    - Run the following commands from the base directory of this repo:
    
```
conda create --name <env> python=3.8.5
conda activate <env>
pip install -e .
```

## Implementation

This repository contains the code of different python functions that estimate the distance between a given pair of chord as well as functions that allow to compare two chord annotations of a given music piece.

Every chord is represented by 
    - a root (int)
    - a chord type (ChordType object)
    - an inversion (int)
    - a pitch type (PitchType object)

The pitch type specifies if the pitches of the chord are represented as MIDI note or as pitch class.

Three metrics are implemented: 

   1. the tone by tone metric: The similarity measure is the number of matches over the maximum possible number of matches between the two chords (The match between roots and basses can be weighted).
       
   2. The voice leading metric: The similarity measure is the number of semitones there are between each pair of pitches in the two chords (The match between basses can be weighted).
       
   3. The Spectral Pitch Simalirity (SPS) metric: The similarity measure is the cosine distance between the two spectrum of each chord. 
