#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pretty_midi

from scipy.fft import fft
from scipy.spatial import distance

from data_types import ChordType
from utils import get_chord_pitches

def chord_SPS(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    program1: int = 0,
    program2: int = 0
) -> float : 
    """
    Get the spectral pitch similarity (SPS) between two chords (composed of
    a root a ChordType  nd and inversion) using general MIDI programs

    Parameters
    ----------
    root1 : int
        The root of the given first chord, as MIDI note number. If the chord 
        is some inversion, the root pitch will be on this MIDI note, but there 
        may be other pitches below it.
    root2 : int
        The root of the given second chord, as MIDI note number. If the chord 
        is some inversion, the root pitch will be on this MIDI note, but there 
        may be other pitches below it.
    chord_type1 : ChordType
        The chord type of the given first chord.
    chord_type2 : ChordType
        The chord type of the given second chord.
    inversion1 : int, optional
        The inversion of the first chord.
        The default is 0.
    inversion2 : int, optional
        The inversion of the second chord.
        The default is 0.
    program1 : int, optional
        The general MIDI program number used by the fluidsynth to synthesize 
        the wave form of the MIDI data of the first chord (it uses the 
        TimGM6mb.sf2 sound font file included with pretty_midi and synthesizes 
        at fs=44100hz by default)
        
        The general MIDI program number (instrument index) is in [0, 127] : 
        https://pjb.com.au/muscript/gm.html
        
        The default is 0.
    program2 : int, optional
        The general MIDI program number used by the fluidsynth to synthesize 
        the wave form of the MIDI data of the second chord.
        The default is 0.

    Returns
    -------
    float 
        The cosin distance beween the spectra of the two synthesized chords (SPS).
        (in [0, 1])
    """
    ## MIDI data of the frist chord :
    
    pm1 = pretty_midi.PrettyMIDI() # MIDI object
    instrument1 = pretty_midi.Instrument(program=program1) #Instrument instance
    
    # note number of the pitches in the chord
    notes1 = get_chord_pitches(root = root1,\
                               chord_type = chord_type1,\
                               inversion= inversion1) 
    notes1 += 60 # centered around C4
    
    for note_number in notes1:
        note = pretty_midi.Note(
            velocity=100, pitch=note_number, start=0, end=1) # Note instance
        instrument1.notes.append(note)
        
    pm1.instruments.append(instrument1)
    
    ## MIDI data of the second chord :
    
    pm2 = pretty_midi.PrettyMIDI()
    instrument2 = pretty_midi.Instrument(program=program2)
    
    notes2 = get_chord_pitches(root = root2,\
                               chord_type = chord_type2,\
                               inversion= inversion2) 
    notes2 += 60 
    
    for note_number in notes2:
        note = pretty_midi.Note(
            velocity=100, pitch=note_number, start=0, end=1)
        instrument2.notes.append(note)
        
    pm2.instruments.append(instrument2)

    ## Spectrum of the synthesized wave form of the MIDI data 
    
    y1 = pm1.fluidsynth()
    yf1 = fft(y1)
    yf1 = 2.0/len(yf1) * np.abs(yf1[0:len(yf1)//2])
    
    y2 = pm2.fluidsynth()
    yf2 = fft(y2)
    yf2 = 2.0/len(yf2) * np.abs(yf2[0:len(yf2)//2])
    
    return(distance.cosine(yf1, yf2))