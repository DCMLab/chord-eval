#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pretty_midi

from librosa import stft
from scipy.fft import fft
from scipy.spatial import distance

from data_types import ChordType
from utils import get_chord_pitches


def creat_chord(
    root: int,
    chord_type: ChordType,
    inversion: int = 0,
    program: int = 0
) -> pretty_midi.PrettyMIDI :
    """
    Creat a pretty_midi object with an instrument which has the given chord of
    duration 1s.

    Parameters
    ----------
    root : int
        The root of the given chord, as MIDI note number. If the chord 
        is some inversion, the root pitch will be on this MIDI note, but there 
        may be other pitches below it.    
    chord_type : ChordType
        The chord type of the given chord.    
    inversion : int, optional
        The inversion of the chord. The default is 0.    
    program : int, optional
        The general MIDI program number used by the fluidsynth to synthesize 
        the wave form of the MIDI data of the first chord (it uses the 
        TimGM6mb.sf2 sound font file included with pretty_midi and synthesizes 
        at fs=44100hz by default)
        
        The general MIDI program number (instrument index) is in [0, 127] : 
        https://pjb.com.au/muscript/gm.html
        
        The default is 0.

    Returns
    -------
    pretty_midi.PrettyMIDI
        A pretty_midi object containing the instrument and the chord to play.

    """  
    # MIDI object
    pm = pretty_midi.PrettyMIDI() 
    
    # Instrument instance
    instrument = pretty_midi.Instrument(program=program)
    
    # note number of the pitches in the chord
    notes = get_chord_pitches(root = root,
                              chord_type = chord_type,
                              inversion= inversion) 
    notes += 60 # centered around C4
    
    for note_number in notes:
        # Note instance of 1s
        note = pretty_midi.Note(velocity=100,
                                pitch=note_number,
                                start=0,
                                end=1) 
        instrument.notes.append(note)
        
    pm.instruments.append(instrument)
    
    return pm


def get_dft_from_MIDI(
    midi_object: pretty_midi.PrettyMIDI
) -> np.ndarray:
    """
    Get the discret Fourier transform of the synthesized instrument's notes 
    contained in the MIDI object using fluidsynth.
    
    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        MIDI object that should contain at least one intrument which should
        contain a program and at least one note.

    Returns
    -------
    ndarray
        The discret Fourier transform.

    """
    x = midi_object.fluidsynth()
    
    # dft = fft(x)
    # dft = 2.0/len(dft) * np.abs(dft[0:len(dft)//2])
    
    dft = np.abs(stft(x))
    # middle = int(np.floor(dft.shape[1]/2))
    # dft = dft[:,middle]

    return dft


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
    # MIDI object of the frist chord :
    pm1 = creat_chord(root=root1,
                      chord_type=chord_type1,
                      inversion=inversion1)
    
    # MIDI object of the second chord :
    pm2 = creat_chord(root=root2,
                      chord_type=chord_type2,
                      inversion=inversion2)

    # Spectrum of the synthesized instrument's notes contained in the MIDI
    # object :
    dft1 = get_dft_from_MIDI(pm1)
    dft2 = get_dft_from_MIDI(pm2)
    
    dist_list = []
    for i in range(dft1.shape[1]):
        dist = distance.cosine(dft1[:,i], dft2[:,i])
        dist_list.append(dist)
        
    dist = np.mean(dist_list)  
    #d = distance.cosine(dft1, dft2)
    
    return(dist)