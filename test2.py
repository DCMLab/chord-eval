#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:01:04 2020

@author: xavier
"""

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import fluidsynth

from scipy.fft import fft

###

# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
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

y = cello_c_chord.fluidsynth()

###

yf = fft(y)
N = len(yf)
sr=44100

yf = 2.0/N * np.abs(yf[0:N//2])
xf = np.linspace(0.0, sr/2.0, N//2)

plt.plot(xf[:2000], yf[:2000])
plt.grid()
plt.show()

