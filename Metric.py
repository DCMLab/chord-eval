#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import pretty_midi

from librosa import stft, cqt, vqt
from librosa.feature import melspectrogram
from scipy.signal import medfilt
from scipy.spatial import distance
from collections import defaultdict

from data_types import ChordType, PitchType
from utils import get_chord_pitches
from constants import TRIAD_REDUCTION



def creat_chord(
    root: int,
    chord_type: ChordType,
    inversion: int = 0,
    program: int = 0
) -> pretty_midi.PrettyMIDI :
    """
    Creat a pretty_midi object with an instrument and the given chord for a 1s 
	duration.

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
        
        The default is 0 for piano.

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
							  pitch_type = PitchType.MIDI,
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
    n_mels: int = 512
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
    Filter the discret fourier transform passed in argument : For each 
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


def spectrogram_caching(
    root: int,
    chord_type: ChordType,
    inversion: int = 0,
    program: int = 0,
    transform: str = 'vqt',
    hop_length: int = 512,
    bins_per_octave: int = 60,
    n_mels: int = 512,
    spectrogram: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
) -> dict:
    """
    Store the spectrogram of the chord in a dictonary.
    
     Parameters
    ----------
    root : int
        The root of the given chord, as MIDI note number. If the chord 
        is some inversion, the root pitch will be on this MIDI note, but there 
        may be other pitches below it.
        
    chord_type : ChordType
        The chord type of the given chord.
        
    inversion : int, optional
        The inversion of the chord.
        The default is 0.
        
    program : int, optional
        The general MIDI program number used by the fluidsynth to synthesize 
        the wave form of the MIDI data of the given chord (it uses the 
        TimGM6mb.sf2 sound font file included with pretty_midi and synthesizes 
        at fs=44100hz by default)
        The general MIDI program number (instrument index) is in [0, 127] : 
        https://pjb.com.au/muscript/gm.html
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
        
    spectrogram : dict, optional
        A dict containing the spectrograms of the chords.
        The default is 
		defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    
    Returns
    -------
    dict
        The dictionary that store the spectrogram of the synthezised chord.
    """
    # MIDI object of the chord :
    pm = creat_chord(root=root,
                      chord_type=chord_type,
                      inversion=inversion,
                      program=program)
  
    # Second instance of the chord an octave below the fisrt one.
    pm_below = creat_chord(root=root - 12,
                            chord_type=chord_type,
                            inversion=inversion,
                            program=program)
    
    # Third instance of the chord an octave above the fisrt one.
    pm_above = creat_chord(root=root + 12,
                            chord_type=chord_type,
                            inversion=inversion,
                            program=program)
    
    
    # Spectrum of the synthesized instrument's notes of the MIDI object :
    dft = get_dft_from_MIDI(pm, 
                            transform=transform,
                            hop_length=hop_length, 
                            bins_per_octave=bins_per_octave,
                            n_mels=n_mels)
    
    dft_below = get_dft_from_MIDI(pm_below, 
                                  transform=transform,
                                  hop_length=hop_length, 
                                  bins_per_octave=bins_per_octave,
                                  n_mels=n_mels)
    
    dft_above = get_dft_from_MIDI(pm_above, 
                                  transform=transform,
                                  hop_length=hop_length, 
                                  bins_per_octave=bins_per_octave,
                                  n_mels=n_mels)
    
    
    # Add the spectrograms to the dict
    try :
        spectrogram[transform][program][root][chord_type][inversion] = [dft_below,
                                                                        dft,
                                                                        dft_above]
        
    except KeyError :
        spectrogram = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        spectrogram[transform][program][root][chord_type][inversion] = [dft_below,
                                                                        dft,
                                                                        dft_above]
        
    return spectrogram

    

def SPS_distance(
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
    n_mels: int = 512,
    noise_filtering: bool = False,
    peak_picking: bool = False,
    spectrogram: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
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
    
    transform : str, optional
        Indicates which transform to use to get the spectrogram of the 
        synthesized instrument's notes. It can either be 'stft, 'cqt', 'vqt' 
        or 'mel'/'melspectrogram'.
        The default is 'vqt'
    
    hop_length : int, optional
        Number of samples between successive CQT or VQT columns if transform is
        'cqt' or 'vqt'.
        the default is 512.
        
    bin_per_octave : int, optional
        Number of bins per octave if transform is 'cqt' or 'vqt'.
        the default is 60.
        
    n_mels : int, optional
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
        The default is defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    
    Returns
    -------
    float 
        The cosin distance beween the spectra of the two synthesized chords
        (SPS) (in [0, 1]).
		
	dict
        The dictionary that store the spectrogram of the synthezised chord.
    """
    try : 
        dft1_below, dft1, dft1_above = spectrogram[transform][program1][root1][chord_type1][inversion1]
        
    except KeyError :
        spectrogram = spectrogram_caching(root=root1,
                                          chord_type=chord_type1,
                                          inversion=inversion1,
                                          program=program1,
                                          transform=transform,
                                          hop_length=hop_length, 
                                          bins_per_octave=bins_per_octave,
                                          n_mels=n_mels,
                                          spectrogram=spectrogram)
        
        dft1_below, dft1, dft1_above = spectrogram[transform][program1][root1][chord_type1][inversion1]
    
        
    try : 
        dft2_below, dft2, dft2_above = spectrogram[transform][program2][root2][chord_type2][inversion2]
        
    except KeyError :
        spectrogram = spectrogram_caching(root=root2,
                                          chord_type=chord_type2,
                                          inversion=inversion2,
                                          program=program2,
                                          transform=transform,
                                          hop_length=hop_length, 
                                          bins_per_octave=bins_per_octave,
                                          n_mels=n_mels,
                                          spectrogram=spectrogram)
        
        dft2_below, dft2, dft2_above = spectrogram[transform][program2][root2][chord_type2][inversion2]

    
    if peak_picking or noise_filtering : 
        dft1 = filter_noise(dft1)
        dft1_below = filter_noise(dft1_below)
        dft1_above = filter_noise(dft1_above)
        
        dft2 = filter_noise(dft2)
        dft2_below = filter_noise(dft2_below)
        dft2_above = filter_noise(dft2_above)
        
        if peak_picking : 
            dft1 = find_peaks(dft1)
            dft1_below = find_peaks(dft1_below)
            dft1_above = find_peaks(dft1_above)
            
            dft2 = find_peaks(dft2)
            dft2_below = find_peaks(dft2_below)
            dft2_above = find_peaks(dft2_above)
    
    # SPS
    dist = [distance.cosine(dft1_below, dft2_below)]
    dist.append(distance.cosine(dft1_below, dft2))
    dist.append(distance.cosine(dft1_below, dft2_above))
    
    dist.append(distance.cosine(dft1, dft2_below))
    dist.append(distance.cosine(dft1, dft2))
    dist.append(distance.cosine(dft1, dft2_above))
    
    dist.append(distance.cosine(dft1_above, dft2_below))
    dist.append(distance.cosine(dft1_above, dft2))
    dist.append(distance.cosine(dft1_above, dft2_above))
    
    # Return the smaller distance
    dist = min(dist)

    return dist, spectrogram


#%%

def compare_notes(
    note1: int,
    note2: int
) -> (int, int):
    """
    Find the smallest interval between tow given midi notes by shifting them by 
    one or more octaves and return the shifted notes.

    Parameters
    ----------
    note1 : int
        Midi number of the first note.
    note2 : int
        Midi number of the second note

    Returns
    -------
    (int, int)
        Midi numbers of the two shiftd notes.

    """
    if note1 - note2 < 0 :
        while np.abs(note1 - note2 + 12) <= np.abs(note1 - note2):
            note1 += 12
            
    else :
        while np.abs(note1 - note2 - 12) <= np.abs(note1 - note2):
            note2 += 12
    
    return note1, note2

def voice_leading_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
	pitch_type: PitchType = PitchType.MIDI,
	only_bass: bool = True,
    bass_weight: int = 1,
    duplicates: bool = False
) -> float:
    """
	   Get the voice leading distance between two chords : the number
    of semitones between the pitches of each chords. 
    
	Parameters
	----------
	root1 : int
		he root of the given first chord, as MIDI note number. If the chord 
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
		 The inversion of the first chord. The default is 0.
		 
	inversion2 : int, optional
		The inversion of the second chord.. The default is 0.
		
	pitch_type : PitchType, optional
		The pitch type of the given root. If PitchType.MIDI, the root is treated
        as a MIDI note number with C4 = 60. If PitchType.TPC, the root is treated
        as an interval above C along the circle of fifths (so G = 1, F = -1, etc.).
		The default is PitchType.MIDI.
		
	only_bass : bool, optional
		If PitchType.TPC, this parameter specify if only the bass is treated with 
		its TPC representation. The default is True.
		
	bass_weight : int, optional
		The weight of the basses distance of the two chords. The default is 1.
		
	duplicates : bool, optional
		For a pair of chords of different lenght, it specifies if the bass of
		the smallest chord (ehich is compare to the bass of the other chord) is 
		duplicated in order to compare against it the other pitches of the other
		chord. 
		The default is False.

	Returns
	-------
	float
		The voice leading distance between two chords.

	"""
    # note number of the pitches in the chord
    notes1 = pd.Series(get_chord_pitches(root = root1,
                                         chord_type = chord_type1,
										 pitch_type = pitch_type,
                                         inversion = inversion1))
    
    notes2 = pd.Series(get_chord_pitches(root = root2,
                                         chord_type = chord_type2,
										 pitch_type = pitch_type,
                                         inversion = inversion2))
    
	# Matrix that keep the number of semi-tones between each pair of pitches between 
	# the two chords.
    similarities = np.ndarray((len(notes1), len(notes2)))
    
	#__________________________________________________________________________
	# Bass weighted similarity
    bass1, bass2 = compare_notes(notes1[0],notes2[0])
    bass_steps = bass_weight*np.abs(bass1-bass2)
    
		
	#__________________________________________________________________________
	# Computaion of the smallest overall number of semi-tones between the rest 
	# of the two chord using graph representation
	
    G = nx.Graph()
	
	# Specify if the bass only is in its TPC representation
    if only_bass and pitch_type == PitchType.TPC:
        pitch_type=PitchType.MIDI
		
        notes1 = pd.Series(get_chord_pitches(root = root1,
                                         chord_type = chord_type1,
										 pitch_type = pitch_type,
                                         inversion = inversion1))
    
        notes2 = pd.Series(get_chord_pitches(root = root2,
                                         chord_type = chord_type2,
										 pitch_type = pitch_type,
                                         inversion = inversion2))
    
	# Specify if the bass can be duplicated
    if duplicates :
        for idx1, note1 in notes1.items():
            for idx2, note2 in notes2.items():
            
                note1, note2 = compare_notes(note1, note2) # Find the smallest intervall
														   # between the two pitches
				
                similarity = 6 - np.abs(note1-note2) # 6 being the maximum number 
													 # semi-tones between two pitches
													 
                similarities[idx1,idx2] = similarity 
        
                G.add_weighted_edges_from([(idx1,idx2+len(notes1), similarity)])
    
    else :
        for idx1, note1 in notes1[1:].items():
            for idx2, note2 in notes2[1:].items():
                
                note1, note2 = compare_notes(note1, note2)
                similarity = 6 - np.abs(note1-note2) 
                similarities[idx1,idx2] = similarity
				
				# the second node is define by the length of the first chord plus 
				# the index of the corresponding pitch within the second chord 
                G.add_weighted_edges_from([(idx1,idx2+len(notes1), similarity)])
            
    matching = nx.max_weight_matching(G, maxcardinality=True)
    similarities[0,0] = bass_steps # replace the similarity between the basses in
								   # the similarities matrix by their weighted one
	
	# Tatal distance							          
    total_steps = bass_steps	
    for pair in matching:
        if pair != (0, len(notes1)):
			
			# We substract the length of the first chord to the second node of the
			# matching to the index of the corresponding pitch within the second chord
            total_steps+=6-similarities[pair[0], pair[1]-len(notes1)]
    	
	#__________________________________________________________________________	
	# Tackle different chord length
	
    if len(notes1)!=len(notes2):
		
		# Find which chord is the longest and keep track of its index
        if len(notes1)>len(notes2):
            big_chord = notes1.copy()
            bc_idx = 0
            short_chord = notes2.copy()
        elif len(notes1)<len(notes2):
            big_chord = notes2.copy()
            bc_idx = 1
            short_chord = notes1.copy()
         
		# Remove all the pitches in the biggest chord that have already been 
		# matched (and the bass if we didn't allow it for duplicate)
        if not duplicates :
            big_chord.drop(0, inplace=True)	
        for pair in matching: 
            idx = pair[bc_idx] - len(notes1) if bc_idx else pair[bc_idx]
            big_chord.drop(idx, inplace=True)
          
		# Computaion of the smallest overall number of semi-tones between the rest 
		# of the big chord and the small chord : Each time we find a matching
		# we deleted the new matched pitches of the big chord until it's ampty, 
		# meaning we have found a matching for all its extra notes.
        while (len(big_chord)!=0):
			
			# Looking for  matching between extra note of the big chord and the 
			# small chord, using graph
            G = nx.Graph()
            for idx_b, note_b in big_chord.items():
                for idx_s, note_s in short_chord.items():
                
                    note_b, note_s = compare_notes(note_b, note_s)
                    similarity = 6-np.abs(note_b-note_s)
                    if bc_idx :
                        similarities[idx_s,idx_b] = similarity
                        G.add_weighted_edges_from([(idx_b,idx_s+len(notes2),
                                                    similarity)])
                    else :
                        similarities[idx_b,idx_s] = similarity
                        G.add_weighted_edges_from([(idx_b,idx_s+len(notes1),
                                                    similarity)])        
                
            matching = nx.max_weight_matching(G)
			
            for pair in matching:
				
				# Get rid of the new matched pitches
                big_chord.drop(pair[0], inplace=True)
				
				# Update the total distance
                if bc_idx :
                    total_steps+=6-similarities[pair[1]-len(notes2), pair[0]]
                else :
                    total_steps+=6-similarities[pair[0], pair[1]-len(notes1)]

    return total_steps
    

#%%

def tone_by_tone_distance(
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    root_weight: int = 1,
    bass_weight: int = 1
) -> float:
    """
    Get the tone by tone distance between two chords : the number
    of matched pitches between the chords over the maximum number of matched pitch. 


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
        
    root_weight : int, optional
        The weight of the matching of the two chords' root.
        The default is 1.
        
    base_weight : int, optional
        The weight of the matching of the two chords' bass.
        The default is 1.

    Returns
    -------
    float
        The tone by tone distance between two chords.

    """
    notes1 = get_chord_pitches(root = root1,
                               chord_type = chord_type1,
							   pitch_type = PitchType.MIDI,
                               inversion = inversion1)%12
    
    notes2 = get_chord_pitches(root = root2,
                               chord_type = chord_type2,
							   pitch_type = PitchType.MIDI,
                               inversion = inversion2)%12
    
    matches = sum([1 if note in notes2 else 0 for note in notes1])
    
    if root1 == root2:
        matches += root_weight-1
    if notes1[0] == notes2[0]:
        matches += bass_weight-1
        
    dist = 1-matches/(max(len(notes1), len(notes2))+root_weight+bass_weight-2)
    
    return dist 
    
    

#%%

def get_distance(
    distance: str,
    root1: int,
    root2: int,
    chord_type1: ChordType,
    chord_type2: ChordType,
    inversion1: int = 0,
    inversion2: int = 0,
    triad_reduction: bool = False,
    equivalences: str = None,
    eq_val: float = 0.5,
    **kwargs
) -> float:
    """
    

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
        
    triad_reduction : bool, optional
        If True, the function will reduce every input chords to their triad
        reduction.
        The default is False.
        
    equivalences : str, optional
        Specify which equivalence we take into account. 
		The default is None.
		
    eq_val : float, optional
        Specify which value to return in case of chord equivalence.
		 The default is 0.5.
		 
    **kwargs : TYPE
        Additional argument for the type of metric used.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    float
        The corresponding distance between tĥe two chords.

    """
    if triad_reduction:
        chord_type1 = TRIAD_REDUCTION[chord_type1]
        chord_type2 = TRIAD_REDUCTION[chord_type2]
        
    if not equivalences==None:
        equivalences = equivalences.split(", ")
	        
        if "octave" in equivalences:
	        root1 = root1//12
	        root2 = root2//12
	        
        if "octatonic" in equivalences and root1 %3 == root2 %3 :
	        root2 = root1
	        if chord_type1 == chord_type2 :
	            return eq_val
	            
	            
        if "hexatonic" in equivalences and root1 %4 == root2 %4 :
	        root2 = root1
	        if chord_type1 == chord_type2 :
	            return eq_val
    
    if distance == "SPS":
        return SPS_distance(root1=root1,
                         root2=root2,
                         chord_type1=chord_type1,
                         chord_type2=chord_type2,
                         inversion1=inversion1,
                         inversion2=inversion2,
                         **kwargs)[0]
    
    elif distance == "voice leading":
        return voice_leading_distance(root1=root1,
                                      root2=root2,
                                      chord_type1=chord_type1,
                                      chord_type2=chord_type2,
                                      inversion1=inversion1,
                                      inversion2=inversion2,
                                      **kwargs)
    
    elif distance == "tone by tone":
        return tone_by_tone_distance(root1=root1,
                                     root2=root2,
                                     chord_type1=chord_type1,
                                     chord_type2=chord_type2,
                                     inversion1=inversion1,
                                     inversion2=inversion2,
                                     **kwargs)
	
    elif distance == "binary":
        return 0 if tone_by_tone_distance(root1=root1,
                                     root2=root2,
                                     chord_type1=chord_type1,
                                     chord_type2=chord_type2,
                                     inversion1=inversion1,
                                     inversion2=inversion2,
                                     **kwargs) > 0 else 1
	
    else:
        raise ValueError("distance must be "
                         "'SPS', 'voice leading', 'tone by tone' or 'binary'.")

