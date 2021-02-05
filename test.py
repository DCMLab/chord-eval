#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Chord_SPS_copy import chord_SPS
from data_types import ChordType

spect = {}

d = chord_SPS(0, 0, ChordType.MAJOR, ChordType.MAJOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj and Cmaj is : {:.5}'.format(d))
d = chord_SPS(0, 0, ChordType.MAJOR, ChordType.MAJOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj and Cmaj is : {:.5}'.format(d))

d = chord_SPS(0, 0, ChordType.MAJOR, ChordType.MINOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj and Cmin is : {:.5}'.format(d))
d = chord_SPS(0, 0, ChordType.MAJOR, ChordType.MINOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj and Cmin is : {:.5}'.format(d))

d = chord_SPS(0, 4, ChordType.MAJ_MAJ7, ChordType.MINOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj7 and Emin is : {:.5}'.format(d))
d = chord_SPS(0, 4, ChordType.MAJ_MAJ7, ChordType.MINOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj7 and Emin is : {:.5}'.format(d))

d = chord_SPS(0, 9, ChordType.MAJOR, ChordType.MIN_MIN7, transform='vqt', spectrogram=spect)
print('The SPS of C4maj and A4min7 is : {:.5}'.format(d))
d = chord_SPS(0, 9, ChordType.MAJOR, ChordType.MIN_MIN7, transform='vqt', spectrogram=spect)
print('The SPS of C4maj and A4min7 is : {:.5}'.format(d))

d = chord_SPS(0, -3, ChordType.MAJOR, ChordType.MIN_MIN7, transform='vqt', spectrogram=spect)
print('The SPS of C4maj and A3min7 is : {:.5}'.format(d))
d = chord_SPS(0, -3, ChordType.MAJOR, ChordType.MIN_MIN7, transform='vqt', spectrogram=spect)
print('The SPS of C4maj and A3min7 is : {:.5}'.format(d))

d = chord_SPS(0, 1, ChordType.MAJOR, ChordType.MAJOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj and Dbmaj is : {:.5}'.format(d))
d = chord_SPS(0, 1, ChordType.MAJOR, ChordType.MAJOR, transform='vqt', spectrogram=spect)
print('The SPS of Cmaj and Dbmaj is : {:.5}'.format(d))




