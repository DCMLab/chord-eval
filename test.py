#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Chord_SPS import chord_SPS
from data_types import ChordType


d = chord_SPS(0,4,ChordType.MAJ_MAJ7,ChordType.MINOR)
print('The SPS of Cmaj7 and Emin is : {}'.format(d))
d = chord_SPS(0,-3,ChordType.MAJOR,ChordType.MIN_MIN7)
print('The SPS of Cmaj and Amin7 is : {}'.format(d))
d = chord_SPS(0,0,ChordType.MAJOR,ChordType.MINOR)
print('The SPS of Cmaj and Cmin is : {}'.format(d))