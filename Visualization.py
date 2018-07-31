
# coding: utf-8

# In[10]:


import os
import librosa 
import librosa.display
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import IPython.display
from tensorflow.python.platform import gfile


# In[11]:


PATH_TRAIN='train'


# In[12]:


path=os.path.join(PATH_TRAIN,'*','*.wav')
wav_path=gfile.Glob(path)[3]


# In[13]:


IPython.display.Audio(wav_path)


# In[14]:


wave,sr=librosa.load(wav_path,mono=True)
plt.figure(figsize=(12,4))
librosa.display.waveplot(wave,sr=sr)
plt.title('Amplitude envelop of Waveform ')


# In[15]:


mfccs=librosa.feature.mfcc(y=wave,sr=sr,n_mfcc=20)
plt.figure(figsize=(12,4))
plt.title('MFCC')
librosa.display.specshow(mfccs,x_axis='time')
plt.colorbar()


# In[16]:


mfccs

