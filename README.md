# chord-eval

## packages to install :

Chord_SPS.py : 
  * librosa (https://librosa.org/)
  * pretty_midi (https://craffel.github.io/pretty-midi/)
  * PyFluidSynth (https://github.com/nwhitehead/pyfluidsynth/archive/master.zip)
  * fluidsynth (https://www.fluidsynth.org/)
  


## Implementation of the chord_SPS function :

It gets the spectral pitch similarity (SPS) between two chords (composed
of a root a chord type and an inversion) using general MIDI programs.

### Genreal MIDI program number :

The function chord_SPS needs two GM program number (for the two chords to 
synthesize) from a list given here : https://pjb.com.au/muscript/gm.html
that is given in argument to the creat_chord() function (with the root, 
the chord type and the inversion of each chords).

The creat_chord() function creats a pretty_midi instance (MIDI object) 
with the list of the notes of the chord to synthezise with the GM program
number.

### Octave picking : 

The SPS between the two chords depends on where each chord is palyed.
The function computes thus the SPS between the two chords for different
octaves for the second chord and returns the smallest SPS.
 
### Fourier transform : 

The function use the get_dft_from_MIDI() function that synthezises the 
MIDI data into a time-serie array using FluidSynth and then computes the 
spectrogram of the time-serie using either stft, cqt, vqt or melspectrogram
transformation.

The get_dft_from_MIDI() function returns only the middle frame of the 
spectrogram because it capture the frequency content of the sustained note.

### Noise filtering and peak picking : 

The function can call the filter_noise() function that filter out the 
noise of the dicret fourier transform passed in argument : For each 
component of the spectrum, the median magnitude is calculated across a 
centred window (the fourier transform is zero-padded). If the magnitude
of that component is less that a noise-factor times the window’s median,
it is considered noise and removed.

The function can also call the peak_picking() function that isolate the 
peaks of a spectrum : If consecutive bins in the spectrum are non-zero,
the function keep only the maximum of the bins and filter out the others.

### Caching

A dict can be given in argument to the function for caching : the 
spectrogram of each new chord encontered is sotered in it.



## test.py

A script to test the chord_SPS function

## data_reduc

This folder contains 4 csv files with SPS values computed with the 4 
different spectrogram transformations : stft, cqt, vqt and melspectrogram.

Each of them regroups the SPS value between triades (4 types) in all their
inversions (3 inversions) for 25 roots(from C3 to C5) to the C4maj, C4min,
C4dim and C4aug triades in their different inversions.
Each of the SPS value is computed 3 times using firts no filtering nor
peak picking, then noise filtering and finally noise filtering and peak
picking.

/!\ The cqt file doesn't have the SPS values for every inversion /!\



## comparison.ipynb

A jupyter note book that shows different plots to compare each SPS values
between them.



## output_sps_kse-100

The folder containing the coprus



## SPS_output.py

Script that computes the mean accuracies (binary, SPS, ...) for each 
pieces in the corpus.
