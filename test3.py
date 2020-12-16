#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:08:40 2020

@author: xavier
"""

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import librosa

from midi2audio import FluidSynth
from scipy.fft import fft

###

# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument 
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)
# Iterate over note names, which will be converted to note number later
for note_name in ['C5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=0, end=.5)
    # Add it to our cello instrument
    cello.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)
# Write out the MIDI data
cello_c_chord.write('cello-C-chord.mid')

fs = FluidSynth()
fs.midi_to_audio('cello-C-chord.mid', 'output.wav')

###

y, sr = librosa.load('output.wav')

yf = fft(y)
N = len(yf)

yf = 2.0/N * np.abs(yf[0:N//2])
xf = np.linspace(0.0, sr/2.0, N//2)

plt.plot(xf, yf)
plt.grid()
plt.show()