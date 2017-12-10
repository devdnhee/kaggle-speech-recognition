
# coding: utf-8

# # Data exploration Kaggle Speech Recognition

# In[1]:


import os.path
from os import listdir
from scipy.io import wavfile
import pyaudio as audio
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import hashlib
import numpy as np


# Defining some constants

# In[2]:


# All words to be recognized in the competition
CORE_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 
              'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
# Directory name of background noise
BACKGROUND_NOISE = '_background_noise_'
# 'unknown' label
UNKNOWN = 'unknown'
# filename for the .csv with all filepaths for easy import
TRAIN_CSV = 'train.csv'
RAW_CSV = 'raw.csv'
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1 # ~134M


# Some handy functions

# In[3]:


def play(sound, samp_freq=16000):
    """play sound, default samp_freq is 16kHz"""
    p = audio.PyAudio()
    stream = p.open(format = audio.paInt16,    
                    rate = samp_freq,
                    channels = 1,
                    output = True)
    stream.write(sound.tobytes())
    stream.close()
    p.terminate()

def which_set(filename, validation_percentage, testing_percentage): 
    """
    Determines which data partition the file should belong to.
    We want to keep files in the same training, validation, or testing sets even if new ones are added over time. This makes it 
    less likely that testing samples will accidentally be reused in training when long runs are restarted for example. 
    To keep this stability, a hash of the filename is taken and used to determine which set it should belong to. 
    This determination only depends on the name and the set proportions, so it won't change as other files are added.
    It's also useful to associate particular files as related (for example words spoken by the same person), 
    so anything after 'nohash' in a filename is ignored for set determination. 
    This ensures that 'bobby_nohash_0.wav' and 'bobby_nohash_1.wav' are always in the same set, for example.
    
    Args: filename: File path of the data sample. validation_percentage: How much of the data set to use for validation. 
        testing_percentage: How much of the data set to use for testing.
        
    Returns: String, one of 'training', 'validation', or 'testing'. 
    """ 
    base_name = os.path.basename(filename) 
    # We want to ignore anything after 'nohash' in the file name when 
    # deciding which set to put a wav in, so the data set creator has a way of 
    # grouping wavs that are close variations of each other. 
    hash_name = re.sub(r'nohash.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should 
    # go into the training, testing, or validation sets, and we want to keep 
    # existing files in the same set even if more files are subsequently 
    # added. 
    # To do that, we need a stable way of deciding based on just the file name 
    # itself, so we do a hash of that and then use that to generate a 
    # probability value that we use to assign it. 
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest() 
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS)) 
    if percentage_hash < validation_percentage: 
        result = 'validation' 
    elif percentage_hash < (testing_percentage + validation_percentage): 
        result = 'testing' 
    else: result = 'training' 
    return result

def random_sample(df, size=1):
    """
    draw random samples from a dataframe
    """
    return df.loc[np.random.choice(df.index, size, replace=True), :]

def read_wavfile(file_path):
    rate, sound = wavfile.read(file_path)
    timestep = np.arange(0.5/rate, 1, 1./rate)
    return dict(zip(timestep, sound))
    


# # Data assemblation
# ## Filepath training dataframe
# Assembling data into a dataframe linking to the path of the soundwaves (which then contains the raw data). The dataframe is exported to `train.csv`.
# Columns of `df`:
# - `speaker`: the id of the participant speaking in the word
# - `word`: the label corresponding to the datapoint (the spoken word)
# - `filepath`: path to the .wav file
# - `utterance`: sequence id of utterance of the word by that speaker

# In[5]:


train_audio_path = os.path.join('train', 'audio')
train_words = [w for w in listdir(train_audio_path) if w != BACKGROUND_NOISE]
dfs = []
columns = ['word', 'filepath']
for w in train_words:
    path = os.path.join(train_audio_path, w)
    sounds = listdir(path)
    #word = w if w in CORE_WORDS else UNKNOWN
    word = w
    records = [(word, os.path.join(path, sound)) for sound in sounds]
    dfs.append(pd.DataFrame.from_records(records, columns=columns))
df = pd.concat(dfs)


# In[6]:


df['speaker'] = df['filepath'].str.split('_').str.get(0).str.split('\\').str.get(-1)
df['utterance'] = df['filepath'].str.split('_').str.get(-1).str.split('.').str.get(0)
df = df.sort_values('speaker')
df.to_csv(TRAIN_CSV, index=False)
df.head()


# ## Raw data DataFrame

# It might be interesting to check some early statistics before actually investigating the waveforms with audio processing analysis

# In[7]:


df = pd.read_csv(TRAIN_CSV)
df['word'] = df['word'].astype('category')


# ### Word distribution

# In[8]:


word_count = df.groupby('word')['filepath'].count()
word_count.plot(kind='bar')
plt.show()


# All words occur approximately a similar amount of time. Due to this, the `unkown` label is much more represented in the dataset than the rest and somewhat unbalanced which might result to a misleading accuracy later on (*accuracy paradox*). There are three approaches:
# - use a logloss as quality measure for the model
# - throw data away to achieve a balanced dataset
# - learn for all different words at the same time
# 
# The last option seems to be the best to me, as we might dodge overfitting with that approach by trying to learn more than we should.
# => change in previous section in code segment `[5]: word = w`.
# 
# Some words occur around 1750 times, whilst others are more in the 2400 range. This is a small imbalance which may be perfectly fine.

# ### Number of speakers

# In[9]:


speakers = df['speaker'].unique()
n_speakers = speakers.shape[0]
print(n_speakers)


# 80% of the speakers will land in the training set, the remaining 20% is for the test set. This ensures an independency between speakers. The same principle will be used for validation later on.

# In[26]:


# some old code to split in sets. Not necessary with the which set function
# pivot = int(0.8 * n_speakers)
# df_reindex = df.set_index('speaker')
# df_train = df_reindex.loc[:speakers[pivot], :]
# df_test = df_reindex.loc[speakers[pivot+1]:, :]

df['set'] = df['filepath'].apply(lambda x: which_set(x, 20., 16.))
df.groupby('set')['speaker'].count()


# ### Utterances of word by same speaker

# Exploring how often words were spoken by the participants

# In[27]:


# generates a table counting how much words were spoken by the speakers
df_speaker_word_count = df.pivot_table(values='filepath', index='speaker', columns='word', aggfunc='count', fill_value=0)
df_speaker_word_count.describe()


# Some words have been uttered over 10 times. The question is then if it is necessary to keep these different utterations? It might be interesting to listen to these outliers in number of utterances, and if needed throw some data away for balancing out.
# Other interesting observations:
# - the median is 1 utterance for every word
# - the 75% quartile is 1 or 2 for every word
# - the 25% quartile is 0 for every word

# In[28]:


df_speaker_word_count2 = df.groupby(['speaker', 'word'])['filepath'].count()
df_speaker_word_count2.hist(bins=12)
plt.show()


# ### Retrieving waveforms

# Reading in the waveform for every file takes quite some time (1GB).

# In[55]:


df['waveform'] = df['filepath'].apply(lambda x: wavfile.read(x)[1])
df.to_csv(RAW_CSV, index=False)


# In[65]:


df = pd.read_csv(RAW_CSV, nrows=5000)
read_wavfile(df.loc['bed', 'filepath'])


# ## Visualization

# In[59]:


rand_w_samples = df.groupby('word', as_index=True).apply(lambda x: random_sample(x, 1)).set_index('word')
#rand_w_samples.loc['bed', 'waveform'].plot()
rand_w_samples.dtypes


# In[63]:


series = df['waveform'].str.strip('[]')
print(series.head())
to_list = lambda x: [-int(d) if d.startswith('-') else int(d) for d in x.split(' ') if d!='']
df['int_waveform'] = series.apply(to_list)
df.head()
df.dtypes


# In[66]:


df.head()


# In[ ]:


read_wavfile(df['filepath'][0])
df2 = pd.DataFrame([read_wavfile(row) for row in df['filepath']])

df3 = pd.DataFrame([read_wavfile(row) for row in df['filepath'][:4])
# In[ ]:


df3.head()

