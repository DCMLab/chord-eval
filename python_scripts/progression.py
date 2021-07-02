#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fractions import Fraction

from metric import get_distance
from data_types import ChordType

PITCH_TO_STRING = {0:'C',1:'Db',2:'D',3:'Eb',4:'E',5:'F',
                   6:'Gb',7:'G',8:'Ab',9:'A',10:'Bb',11:'B'}

def overlap(
    i1:float,
    i2:float
) -> bool:
	"""
	Retrun if the two intervals overlap

	Parameters
	----------
	i1 : float
		first interval : array of two elements.
	i2 : float
		second interval : array of two elements.

	Returns
	-------
	bool
		True if the two intervals overlap.

	"""
	if i1[1]>i2[0] and i1[1]<=i2[1]:
		return True
	if i2[1]>i1[0] and i2[1]<=i1[1]:
		return True
	return False


def duration_overlap(
    i1:float,
    i2:float
) -> float:
	"""
	Return the duration of the common section of the two intervals.

	Parameters
	----------
	i1 : float
		first interval : array of two elements.
	i2 : float
		second interval : array of two elements.

	Returns
	-------
	float
		duration of the common section of the two intervals.

	"""
	if i1[1]>i2[0] and i1[1]<=i2[1]:
		if i1[0]<i2[0]:
			return i1[1]-i2[0]
		else :
			return i1[1]-i1[0]
	if i2[1]>i1[0] and i2[1]<=i1[1]:
		if i2[0]<i1[0]:
			return i2[1]-i1[0]
		else :
			return i2[1]-i2[0]
	return 0



def get_progression(
		df1_path:str,
		df2_path:str
	) -> pd.DataFrame:
	"""
	Construct a pd.DataFrame where each row is a event : a chord change in one of 
	the annotation.

	Parameters
	----------
	df1_path : str
		the path leading to the .tsv file containing the first annotation data.
	df2_path : str
		the path leading to the .tsv file containing the second annotation data.

	Returns
	-------
	progression : TYPE
		The data frame containing the progression of the piece with the two annotations
		and the distances between each annotation.

	"""
	# Load the two annotaions
	
	df1 = pd.read_csv(df1_path, sep='\t', converters={'duration': Fraction})
	df1['chord_type'] = df1['chord_type'].apply(lambda r : ChordType[r.split(".")[1]])
	
	df1['full_chord'] = df1.apply(lambda r : PITCH_TO_STRING[r.chord_root_midi] + '_' + str(r.chord_type).split(".")[1]+"_inv"+str(r.chord_inversion), axis=1)
	
	time_df1 = df1.duration.cumsum().astype(float, copy=False)
	df1['interval'] = [[i, f] for i, f in zip([0]+list(time_df1[:-1]), time_df1)]
    
    ##
    
	df2 = pd.read_csv(df2_path, sep='\t', converters={'duration': Fraction})
	df2['chord_type'] = df2['chord_type'].apply(lambda r : ChordType[r.split(".")[1]])
	
	df2['full_chord'] = df2.apply(lambda r : PITCH_TO_STRING[r.chord_root_midi] + '_' + str(r.chord_type).split(".")[1]+"_inv"+str(r.chord_inversion), axis=1)    
    
	time_fh = df2.duration.cumsum().astype(float, copy=False)
	df2['interval'] = [[i, f] for i, f in zip([0]+list(time_fh[:-1]), time_fh)]

    ##
    
	sps = []
	vl  = []
	tbt = []
	binary = []
	
	time = []
	
	chords_df2 = []
	chords_df1 = []

	idx_df1 = 0
	for idx_df2, rdf2 in df2.iterrows():
		matched_idx = []
		matched_duration = []
		chords_sps_dist = []
		chords_vl_dist = []
		chords_tbt_dist = []
        
		if (idx_df1 > 0 and overlap(df1.interval[idx_df1-1], rdf2.interval)):
            
			matched_idx.append(idx_df1-1)
            
			matched_duration.append(duration_overlap(df1.interval[idx_df1-1], rdf2.interval))
            
			chords_sps_dist.append(get_distance(distance = 'SPS',
                                               root1=df1.chord_root_midi[idx_df1-1],
                                               root2=rdf2.chord_root_midi,
                                               chord_type1=df1.chord_type[idx_df1-1],
                                               chord_type2=rdf2.chord_type,
                                               inversion1=df1.chord_inversion[idx_df1-1],
                                               inversion2=rdf2.chord_inversion))

			chords_vl_dist.append(get_distance(distance = 'voice leading',
                                               root1=df1.chord_root_midi[idx_df1-1],
                                               root2=rdf2.chord_root_midi,
                                               chord_type1=df1.chord_type[idx_df1-1],
                                               chord_type2=rdf2.chord_type,
                                               inversion1=df1.chord_inversion[idx_df1-1],
                                               inversion2=rdf2.chord_inversion,
                                               bass_weight = 3))

			chords_tbt_dist.append(get_distance(distance = 'tone by tone',
                                               root1=df1.chord_root_midi[idx_df1-1],
                                               root2=rdf2.chord_root_midi,
                                               chord_type1=df1.chord_type[idx_df1-1],
                                               chord_type2=rdf2.chord_type,
                                               inversion1=df1.chord_inversion[idx_df1-1],
                                               inversion2=rdf2.chord_inversion,
                                               bass_weight = 3,
                                               root_weight = 3))

		while(idx_df1 < len(df1) and overlap(df1.interval[idx_df1], rdf2.interval)):
            
			matched_idx.append(idx_df1)
            
			matched_duration.append(duration_overlap(df1.interval[idx_df1], rdf2.interval))
            
			chords_sps_dist.append(get_distance(distance = 'SPS',
                                               root1=df1.chord_root_midi[idx_df1],
                                               root2=rdf2.chord_root_midi,
                                               chord_type1=df1.chord_type[idx_df1],
                                               chord_type2=rdf2.chord_type,
                                               inversion1=df1.chord_inversion[idx_df1],
                                               inversion2=rdf2.chord_inversion))

			chords_vl_dist.append(get_distance(distance = 'voice leading',
                                               root1=df1.chord_root_midi[idx_df1],
                                               root2=rdf2.chord_root_midi,
                                               chord_type1=df1.chord_type[idx_df1],
                                               chord_type2=rdf2.chord_type,
                                               inversion1=df1.chord_inversion[idx_df1],
                                               inversion2=rdf2.chord_inversion,
                                               bass_weight = 3))

			chords_tbt_dist.append(get_distance(distance = 'tone by tone',
                                               root1=df1.chord_root_midi[idx_df1],
                                               root2=rdf2.chord_root_midi,
                                               chord_type1=df1.chord_type[idx_df1],
                                               chord_type2=rdf2.chord_type,
                                               inversion1=df1.chord_inversion[idx_df1],
                                               inversion2=rdf2.chord_inversion,
                                               bass_weight = 3,
                                               root_weight = 3))

			idx_df1 += 1
		
		dist_bin = [0 if tbt == 0 else 1 for tbt in chords_tbt_dist]
	    
		sps+=chords_sps_dist
		vl +=chords_vl_dist
		tbt+=chords_tbt_dist
		binary+=dist_bin
	    
		if len(matched_idx)>0:
	        
			time.append(rdf2.interval[0])
			if len(matched_idx)>1:
				for duration in matched_duration:
					time.append(rdf2.interval[0]+duration)
				del time[-1]
	        
			chords_df2 += [rdf2.full_chord]*len(matched_idx)
			chords_df1 += [df1.full_chord[idx] for idx in matched_idx]
            
	progression = pd.DataFrame({'time':time,
	                            'annotation1_chord':chords_df1,
	                            'annotation2_chord':chords_df2,
	                            'sps':sps,
	                            'vl':vl,
	                            'tbt':tbt,
	                            'binary':binary})
	
	return progression



def plot_comparison(
    progression:pd.DataFrame,
    rge:float=None,
    verbose:bool=False,
	title:str=None
	): 
	"""
	Plot the progression of 2 annotaions

	Parameters
	----------
	progression : pd.DataFrame
		pd.DataFrame containg progressino the two annotaion and their distance for
		each chord.
	rge : float, optional
		the range in terms of whole note of the progression : a 2-element array. 
		The default is None.
	verbose : bool, optional
		if True the name of each chord of the two annotation will be marked under 
		the dot of the event. The default is False.
	title : str, optional
		Title of the plot. The default is None.

	Returns
	-------
	None.

	"""
	if rge==None:
		 rge=[progression.time[0],progression.time.iloc[-1]]

	fig, axs = plt.subplots(2,1,figsize=(22,10), sharex=True)

	sns.scatterplot(data=progression.query('time>=@rge[0] and time<@rge[1]'), x='time', y='sps',ax=axs[0], label='SPS')
	sns.scatterplot(data=progression.query('time>=@rge[0] and time<@rge[1]'), x='time', y='tbt',ax=axs[0], label='tone by tone')
	axs[0].grid()
	axs[0].set(xlabel='time', ylabel='distance to matched chord')

	sns.scatterplot(data=progression.query('time>=@rge[0] and time<@rge[1]'), x='time', y='vl',ax=axs[1], label='voice leading')
	axs[1].grid()
	axs[1].set(xlabel='time', ylabel='distance to matched chord')

	if verbose:
        
		limit1=progression.query('time>=@rge[0] and time<@rge[1]').index[0]
		limit2=progression.query('time>=@rge[0] and time<@rge[1]').index[-1]
        
		plt.text(progression.iloc[limit1].time, progression.iloc[limit1].vl-0.5, progression.iloc[limit1].fh_chord, horizontalalignment='center', verticalalignment='top', size='small', fontstretch ='normal', color='maroon')
		for line in range(limit1+1,limit2):
			 if progression.fh_chord[line] != progression.fh_chord[line-1]:
				 plt.text(progression.time[line], progression.vl[line]-0.5, progression.fh_chord[line], horizontalalignment='center', verticalalignment='top', size='small', fontstretch ='normal', color='maroon')

		plt.text(progression.iloc[limit1].time, progression.iloc[limit1].vl+0.5, progression.iloc[limit1].dcml_chord, horizontalalignment='center', verticalalignment='bottom', size='small', fontstretch ='normal')
		for line in range(limit1+1,limit2):
			if progression.dcml_chord[line] != progression.dcml_chord[line-1]:
				plt.text(progression.time[line], progression.vl[line]+0.5, progression.dcml_chord[line], horizontalalignment='center', verticalalignment='bottom', size='small', fontstretch ='normal')


	if not title==None:
		fig.suptitle(title);
	fig.tight_layout()