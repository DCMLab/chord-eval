# chord-eval

## Setting up the environment:

### Download [FluidSynth](https://www.fluidsynth.org):

FluidSynth is needed to get the SPS distance of two chord. It allows to synthesize chords to get their spectrum.

To download it, run the following command if you using Ubuntu or Debian:

```
sudo apt-get install fluidsynth
```

For other OS, clic [here](https://github.com/FluidSynth/fluidsynth/wiki/Download) to see the downloading procedure.

### Conda environment and python packages:

Here are the steps to build a new conda environment with the needed python packages :
    - Download the 2 text files package_conda.txt and requirements.txt
    - Run the following command:
	
```
conda create --name <env> --file /path/to/package_conda.txt
sudo pip install -r requirements.txt
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
       
   3. The Spectral Pitche Simalirity (SPS) metric: The similarity measure is the cosine distance between the two spectrum of each chord. 


## Repository’s folders: 

This Repository contains, three folders:

   - python_scripts: contains the python scripts of the different functions
   - NoteBooks: contains jupyter notebooks that present the features of the different functions and few applications as well as notebooks used to create data files.
   - Data: contains the files used by the different jupyter notebooks.  


### NoteBooks:

The NoteBooks repository contains 8 notebooks:

*metrics_features.ipynb*: The aim of this notebook is to show the properties of the different metrics that have been implemented to estimate the similarity between two chords, and how their output changes with the input parameters.

*sonnatas_annotations.ipynb*: In this notebook, different annotations of a same piece, the first movement of Beethoven sonatas, are compared using the different metrics developed : the binary, SPS, voice leading and tone by tone metrics.

*corpus_analysis.ipynb*: In this notebook a corpus analysis is conducted. The chord progression of several pieces of a corpus has been evaluated by a model so that each pieces has a set of ground truth chords and a set of estimated chords. Each pair of chords are compared using the different metrics.

*automatic_chord_evaluation.ipynb*: The aim of this notebook is to compare different data set of an automatic chord evaluation algorithm with a given ground truth annotation. The different outputs are obtained with different input parameter values.

And the notebooks *data_MF.ipynb*, *data_SA.ipynb*, *data_CA.ipynb*, *data_ACE.ipynb* used to build every csv files needed for the notebooks ‘metrics_features’, ‘sonatas_annotations’, ‘coprus_analysis’ and ‘automatic_chord_evaluation’ respectively.
