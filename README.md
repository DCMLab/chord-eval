# chord-eval

## Implementation

This repository contains the code of different python functions that estimate the distance between a given pair of chord as well as functions that allow to compare two chord annotations of a given music piece.

Every chord is represented by :
    • a root (int)
    • a chord type (ChordType object)
    • an inversion (int)
    • a pitch type (PitchType object)

The pitch type specifies if the pitches of the chord are represented as MIDI note or as pitch class.

Three metric are implemented : 

    1. the tone by tone metric : The similarity measure is the number of matches over the maximum possible number of matches between the two chords (The match between roots and basses can be weighted).
       
    2. The voice leading metric : The similarity measure is the number of semitones there are between each pair of pitches in the two chords (The match between basses can be weighted).
       
    3. The Spectral Pitche Simalirity (SPS) metric : The similarity measure is the cosine distance between the two spectrum of each chord. 


## Repository’s folders : 

This Repository contains, three folder :

    • python_scripts : contains the python scripts of the different functions
    • NoteBooks : contains jupyter notebooks that present the features of the different functions and few applications as well as notebooks used to create data files.
    • Data : contains the files used by the different jupyter notebooks.


## Conda environment and python packages :

Here are the steps to build a new conda environment with the needed python packages :
    1. Download the text file package_conda.txt
    2. Run the following command :
	
	$ conda create --name <env> --file /path/to/package_conda.txt


## Download FluidSynth (https://www.fluidsynth.org/) :

FluidSynth is needed to get the SPS distance of two chord. It allows to synthesize chords to get their spectrum.

To download it, run the following command if you using Ubuntu or Debian:

	$ sudo apt-get install fluidsynth

or go to https://github.com/FluidSynth/fluidsynth/wiki/Download to see the downloading procedure for other OS.
