#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pretty_midi

from librosa import stft, cqt, vqt
from librosa.feature import melspectrogram
from scipy.signal import medfilt
from scipy.spatial import distance
from collections import defaultdict

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
    midi_object: pretty_midi.PrettyMIDI,
    transform: str = 'vqt',
    hop_length: int = 512,
    bins_per_octave: int = 60,
    n_mels : int = 512
) -> np.ndarray:
    """
    Get the discret Fourier transform of the synthesized instrument's notes 
    contained in the MIDI object using fluidsynth.
    
    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        MIDI object that should contain at least one intrument which should
        contain a program and at least one note.
    transform : str
        Indicates which transform to use to get the spectrogram of the 
        synthesized instrument's notes. It can either be 'stft, 'cqt', 'vqt' 
        or 'mel'/'melspectrogram'.
        The default is 'vqt'
    hop_length : int
        Number of samples between successive CQT or VQT columns if transform is
        'cqt' or 'vqt'.
        the default is 512.
    bin_per_octave : int
        Number of bins per octave if transform is 'cqt' or 'vqt'.
        the default is 60.
    n_mels : int
        Number of Mel bands to generate if transform is 'mel' or 'melspectrogram'.
        the default is 512.
    
    Returns
    -------
    ndarray
        The discret Fourier transform.

    """
    sr = 22050 # Librosa's default sampling rate
    
    y = midi_object.fluidsynth(fs=sr) # The sampling rate has to match 
                                         # Librosa's default sampling rate  
                                                                           
                                 
    y = y[:sr]  # fuidsynth sythesizes the 1s-duration note and its release 
                # for an additional second. One keep the first second of the 
                # signal.
    if transform == 'stft' :
        dft = np.abs(stft(y))
    elif transform == 'cqt' :
        dft = np.abs(cqt(y, hop_length=hop_length,
                         n_bins=bins_per_octave * 7,
                         bins_per_octave=bins_per_octave))
    elif transform == 'vqt' :
        dft = np.abs(vqt(y, hop_length=hop_length, 
                         n_bins=bins_per_octave * 7, 
                         bins_per_octave=bins_per_octave))
    elif transform == 'mel' or transform == 'melspectrogram' :
        dft = np.abs(melspectrogram(y, n_mels=n_mels))
    else:
        raise ValueError("transform must be "
                         "'stft', 'cqt', 'vqt', 'mel' or 'melspectrogram' ")
    
    # Return only the middle frame of the spectrogram 
    middle = int(np.floor(dft.shape[1]/2))
    dft = dft[:,middle]

    return dft


def filter_noise(
    dft: np.ndarray,
    size_med: int=41,
    noise_factor: float=4.0,
) -> np.ndarray: 
    """
    Filter the dicret fourier transform passed in argument : For each 
    component of the spectrum, the median magnitude is calculated across a 
    centred window (the fourier transform is zero-padded). If the magnitude
    of that component is less that a noise-factor times the window’s median,
    it is considered noise and removed.
    
    Parameters
    ----------
    dft : np.ndarray
        The discret fourier transform.
    size_med : int
        The width of the centerd window over which the median is calculated
        for each component of the spectrum. The default is 41 bins.
    noise_factor : float
        The noise threshold. The default is 4.0.

    Returns
    -------
    The filtered discret fourier transform

    """
    noise_floor = medfilt(dft,size_med)
    dft_filtered = [0 if x < noise_factor*med else x\
                    for x, med in zip(dft, noise_floor)]
        
    return dft_filtered


def find_peaks(
    dft: np.ndarray
) -> np.ndarray:
    """
    Isolate the peaks of a spectrum : If consecutive bins in the spectrum are 
    non-zero, the function keep only the maximum of the bins and filter the 
    others.
    Parameters
    ----------
    dft : np.ndarray
        The discret fourier transform.

    Returns
    -------
    The filtered discret fourier transform.

    """
    dft_binary = np.append(np.append([0], dft), [0])
    dft_binary = [1 if x > 0 else x for x in dft_binary]
    dft_binary_diff = np.diff(dft_binary)
    
    peaks_start_idx = [idx for idx, diff in enumerate(dft_binary_diff)\
                      if diff==1]
    peaks_end_idx = [idx for idx, diff in enumerate(dft_binary_diff)\
                    if diff==-1]

    peak_idx = [idx+start for start, end in zip (peaks_start_idx,peaks_end_idx)\
                for idx, peak in enumerate(dft[start:end-1])\
                if peak==max(dft[start:end-1])]
    
    peaks = np.array([x if idx in peak_idx else 0\
                  for idx, x in enumerate(dft)])
        
    return peaks

    
def chord_SPS(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    program1: int = 0,
    program2: int = 0,
    transform: str = 'vqt',
    hop_length: int = 512,
    bins_per_octave: int = 60,
    n_mels : int = 512,
    noise_filtering: bool = False,
    peak_picking: bool = False,
    spectrogram: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
) -> float : 
    """
    Get the spectral pitch similarity (SPS) between two chords (composed of
    a root a ChordType and an inversion) using general MIDI programs.

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
    
    transform : str
        Indicates which transform to use to get the spectrogram of the 
        synthesized instrument's notes. It can either be 'stft, 'cqt', 'vqt' 
        or 'mel'/'melspectrogram'.
        The default is 'vqt'
    
    hop_length : int
        Number of samples between successive CQT or VQT columns if transform is
        'cqt' or 'vqt'.
        the default is 512.
        
    bin_per_octave : int
        Number of bins per octave if transform is 'cqt' or 'vqt'.
        the default is 60.
        
    n_mels : int
        Number of Mel bands to generate if transform is 'mel' or 'melspectrogram'.
        the default is 512.
    
    noise_filtering : bool, optional
        If True, the function will filter out the noise of the two spectrums 
        using the filter_noise function.
        If peak_picking is True, it will automatically assign the True value 
        to noise_filtering.
        The default is False.
    
    peak_picking : bool, optional
        If True, the function will isolate the peaks of each spectrum using 
        the find_peaks function after filtering out the noise of each spectrum.
        The default is False.
        
    spectrogram : dict, optional
        A dict containing the spectrograms of the chords.
        The default is None
    
    
    Returns
    -------
    float 
        The cosin distance beween the spectra of the two synthesized chords
        (SPS) (in [0, 1]).
    """
    try : 
        dft1_below, dft1, dft1_above = spectrogram[program1][root1][chord_type1][inversion1]
        
    except KeyError :
    
        # MIDI object of the frist chord :
        pm1 = creat_chord(root=root1,
                          chord_type=chord_type1,
                          inversion=inversion1,
                          program=program1)
  
        # Creat a second instance of the first chord an octave below the fisrt one.
        pm1_below = creat_chord(root=root1 - 12,
                                chord_type=chord_type1,
                                inversion=inversion1,
                                program=program1)
        
        # Creat a third instance of the first chord an octave above the fisrt one.
        pm1_above = creat_chord(root=root1 + 12,
                                chord_type=chord_type1,
                                inversion=inversion1,
                                program=program1)
        
        # Spectrum of the synthesized instrument's notes of the MIDI object :
        dft1 = get_dft_from_MIDI(pm1, transform,
                                 hop_length=hop_length, 
                                 bins_per_octave=bins_per_octave,
                                 n_mels=n_mels)
        
        dft1_below = get_dft_from_MIDI(pm1_below, transform,
                                       hop_length=hop_length, 
                                       bins_per_octave=bins_per_octave,
                                       n_mels=n_mels)
        
        dft1_above = get_dft_from_MIDI(pm1_above, transform,
                                       hop_length=hop_length, 
                                       bins_per_octave=bins_per_octave,
                                       n_mels=n_mels)
        
        # Add the spectrograms to the dict
        try :
            spectrogram[program1][root1][chord_type1][inversion1] = [dft1_below,
                                                           dft1,
                                                           dft1_above]
            
        except KeyError :
            spectrogram = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            spectrogram[program1][root1][chord_type1][inversion1] = [dft1_below,
                                                           dft1,
                                                           dft1_above]
        
    try : 
        dft2_below, dft2, dft2_above = spectrogram[program2][root2][chord_type2][inversion2]
        
    except KeyError :
             
        # MIDI object of the second chord :
        pm2 = creat_chord(root=root2,
                          chord_type=chord_type2,
                          inversion=inversion2,
                          program=program2)
        
        # Creat a second instance of the second chord an octave below the fisrt one.
        pm2_below = creat_chord(root=root2 - 12,
                                chord_type=chord_type2,
                                inversion=inversion2,
                                program=program2)
        
        # Creat a third instance of the second chord an octave above the fisrt one.
        pm2_above = creat_chord(root=root2 + 12,
                                chord_type=chord_type2,
                                inversion=inversion2,
                                program=program2)
    
        
        # Spectrum of the synthesized instrument's notes of the MIDI object :
        dft2 = get_dft_from_MIDI(pm2, transform,
                                 hop_length=hop_length, 
                                 bins_per_octave=bins_per_octave,
                                 n_mels=n_mels)
        
        dft2_below = get_dft_from_MIDI(pm2_below, transform,
                                       hop_length=hop_length, 
                                       bins_per_octave=bins_per_octave,
                                       n_mels=n_mels)
        
        dft2_above = get_dft_from_MIDI(pm2_above, transform,
                                       hop_length=hop_length, 
                                       bins_per_octave=bins_per_octave,
                                       n_mels=n_mels)

        # Add the spectrograms to the dict
        try :
            spectrogram[program2][root2][chord_type2][inversion2] = [dft2_below,
                                                           dft2,
                                                           dft2_above]
            
        except KeyError :
            spectrogram = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            spectrogram[program2][root2][chord_type2][inversion2] = [dft2_below,
                                                           dft2,
                                                           dft2_above]

    
    if peak_picking or noise_filtering : 
        dft1 = filter_noise(dft1)
        dft1_below = filter_noise(dft1_below)
        dft1_above = filter_noise(dft1_above)
        
        dft2 = filter_noise(dft2)
        dft2_below = filter_noise(dft2_below)
        dft2_above = filter_noise(dft2_above)
        
        if peak_picking : 
            dft1 = find_peaks(dft1)
            dft1_below = filter_noise(dft1_below)
            dft1_above = filter_noise(dft1_above)
            
            dft2 = find_peaks(dft2)
            dft2_below = filter_noise(dft2_below)
            dft2_above = filter_noise(dft2_above)
    
    # SPS
    dist = [distance.cosine(dft1_below, dft2)]
    dist.append(distance.cosine(dft1_below, dft2_below))
    dist.append(distance.cosine(dft1_below, dft2_above))
    
    dist.append(distance.cosine(dft1, dft2))
    dist.append(distance.cosine(dft1, dft2_below))
    dist.append(distance.cosine(dft1, dft2_above))
    
    dist.append(distance.cosine(dft1_above, dft2))
    dist.append(distance.cosine(dft1_above, dft2_below))
    dist.append(distance.cosine(dft1_above, dft2_above))
    
    # Return the smaller distance
    dist = min(dist)

    return dist, spectrogram