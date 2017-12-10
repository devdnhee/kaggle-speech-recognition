"""
write all waveform data to a csv file when executing this script
Takes quite some time to finish.
"""

import numpy as np
import pandas as pd
from scipy.io import wavfile
import time

TRAIN_CSV = 'train.csv'
RAW_CSV = 'raw.csv'
TIMESTEP = np.arange(0.5/16000, 1, 1./16000)

def read_wavfile(file_path):
    rate, sound = wavfile.read(file_path)
    return dict(zip(TIMESTEP, sound))

df = pd.read_csv(TRAIN_CSV)
t = time.time()
df2 = pd.DataFrame([read_wavfile(row) for row in df['filepath']])
print(time.time()-t)
df2.to_csv(RAW_CSV)
