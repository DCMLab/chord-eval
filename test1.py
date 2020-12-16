#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:14:36 2020

@author: xavier
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import psonic as ps

from scipy.fft import fft

###

ps.start_recording()

ps.use_synth(ps.PIANO)
ps.play(70)
ps.sleep(0.5)
ps.play(79)

ps.stop_recording()
ps.save_recording('./test1.wav')

###

y, sr = librosa.load('./test1.wav')

yf = fft(y)
N = len(yf)

yf = 2.0/N * np.abs(yf[0:N//2])
xf = np.linspace(0.0, sr/2.0, N//2)

plt.plot(xf, yf)
plt.grid()
plt.show()






