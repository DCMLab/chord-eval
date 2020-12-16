#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:14:36 2020

@author: xavier
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa
import psonic as ps

from scipy.fft import fft

###

file_path = Path('./test1.wav')
abs_filename = str(file_path.absolute())

ps.start_recording()

ps.use_synth(ps.PIANO)
ps.play(70)
ps.sleep(0.5)
ps.play(79)

ps.stop_recording()
ps.save_recording(abs_filename)

###

y, sr = librosa.load(abs_filename)

yf = fft(y)
N = len(yf)

yf = 2.0/N * np.abs(yf[0:N//2])
xf = np.linspace(0.0, sr/2.0, N//2)

plt.plot(xf, yf)
plt.grid()
plt.show()






