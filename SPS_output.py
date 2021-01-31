#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:25:57 2021

@author: xavier
"""

import pandas as pd
import numpy as np

from glob import glob
from fractions import Fraction

from Chord_SPS import chord_SPS
from data_types import ChordType

path_list = []
avg_bin_acc_list = []
avg_root_acc_list = []
avg_triad_acc_list = []
avg_7th_acc_list = []
avg_inv_acc_list = []
avg_sps_list = []

for df_path in glob("output_sps_kse-100/**/*_results.tsv", recursive=True):
    results_df = pd.read_csv(df_path, sep='\t', index_col=0, converters={'duration': Fraction})
        
    results_df['gt_chord_type'] = results_df['gt_chord_type'].apply(lambda r : ChordType[r.split(".")[1]])
    results_df['est_chord_type'] = results_df['est_chord_type'].apply(lambda r : ChordType[r.split(".")[1]])
    
    results_df['sps_distance'] = results_df.apply(lambda r : chord_SPS(r.gt_chord_root,
                                                                       r.est_chord_root,
                                                                       r.gt_chord_type,
                                                                       r.est_chord_type,
                                                                       r.gt_chord_inv,
                                                                       r.est_chord_inv
                                                                      ), axis=1)
    
    path_list.append(df_path)
    avg_bin_acc_list.append(float(np.average(results_df['full_correct'], weights=results_df['duration'])))
    avg_root_acc_list.append(float(np.average(results_df['root_correct'], weights=results_df['duration'])))
    avg_triad_acc_list.append(float(np.average(results_df['triad_correct'], weights=results_df['duration'])))
    avg_7th_acc_list.append(float(np.average(results_df['7th_correct'], weights=results_df['duration'])))
    avg_inv_acc_list.append(float(np.average(results_df['inv_correct'], weights=results_df['duration'])))
    avg_sps_list.append(float(np.average(results_df['sps_distance'], weights=results_df['duration'])))
    
Chord_symbol_recall_df = pd.DataFrame({'path': path_list,
                                       'binary_accuracy' : avg_bin_acc_list,
                                       'root_accuracy' : avg_root_acc_list,
                                       'triad_accuracy' : avg_triad_acc_list,
                                       '7th_accuracy' : avg_7th_acc_list,
                                       'inversion_accuracy' : avg_inv_acc_list,
                                       'average_sps' : avg_sps_list
                                       })

Chord_symbol_recall_df['name'] = Chord_symbol_recall_df.path.apply(lambda r : r[len('output_sps_kse-100/'):len(r)-len('_results.tsv')])

Chord_symbol_recall_df.to_csv('Chord_symbol_recall_df.csv')