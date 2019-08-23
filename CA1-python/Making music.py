
# coding: utf-8

# In[1]:


import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

import scipy.io.wavfile as wav
import IPython.display as ipd


# # Some introductory functions

# ### List comprehensions

# List comprehensions are a versatile syntax for mapping a function (or
# expression) across all elements of a list.

# In[2]:


# The function below accepts a list L
# and returns another list with elements of L three times as large
def three_ize(L):
    # this is an example of a list comprehension
    LC = [3 * x for x in L]
    return LC


# What is the return value of `three_ize(list1)`?

# In[3]:


list1 = [1,4,5,10]


# ### Write the function `scale`

# The function should have the following signature:

# In[5]:


def scale(L, scale_factor):
    '''
    returns a list similar to L, except that each element has been
    multiplied by scale_factor.
    '''  


# ### Write the function `add_2`:

# The function should have the following signature:

# In[8]:


def add_2(L, M):
    '''
    takes two lists L and M and
    returns a single list that is an element-by-element sum of the two arguments
    If the arguments are different lengths, the function add_2 should 
    return a list that is as long as the shorter of the two. 
    Just ignore the extra elements from the longer list.
    '''


# ### Write the function `add_scale_2`:

# The function should have the following signature:

# In[11]:


def add_scale_2(L, M, scale1, scale2):
    '''
    takes two lists L and M and two floating-point numbers L_scale and M_scale. 
    These stand for scale for L and scale for M, respectively.
    Returns a single list that is an element-by-element sum of the two inputs, 
    each scaled by its respective floating-point value. 
    If the inputs are different lengths, your add_scale_2 should return a list that is
    as long as the shorter of the two. Again, just drop any extra elements.
     '''


# ### How can you obtain the same result using numpy arrays?

# ### Helper function `add_noise`

# In[15]:


def add_noise(x, chance_of_replacing, noise):
    """add_noise accepts an original value, x
       and a fraction named chance_of_replacing.

       With the "chance_of_replacing" chance, it
       returns the number x + noise

       Otherwise, it should return x
    """
    r = random.uniform(0, 1)
    if r < chance_of_replacing:
        return x + noise
    else:
        return x


# ### Create the function `array_add_noise`
# Replace entries in a numpy array using the helper function randomize:

def array_add_noise(L,prob,noise):
# Modify the function `array_add_noise` defined above to take an optional parameter `inplace` which by default is True. 
# When `inplace` is False, the function will instead return a numpy array with the modified values, 
# but it won't replace the entries in the original array.

# # Let's start playing with sounds

# ### Create a sinusoidal sound

# In[20]:


# Define the duration of the sound we want to create:
# duration in seconds
duration = 5


# In[21]:


# Define the rate of the sound, which is the number of sample points per second
DEFAULT_RATE = 44100


# In[22]:


# The total number of sample points that define your sound is:
nsamples = int(DEFAULT_RATE*duration)


# In[23]:


# Then create a numpy array that define the range of the sound, i.e., 
# nsamples points equaly spaced in the range (0,duration) [s]
t = np.linspace(0,duration,nsamples)


# Create a sound array corresponding to the function
# 
# $ f(t) = \sin(220 (2 \pi t)) + \sin(224 (2 \pi t))$
data = ...
# Use `plt.plot(t,data)` to plot your function:

# Check the sound you just created!

# In[26]:


#ipd.Audio(data,rate=DEFAULT_RATE)


# You can also try different functions!

# ### Create a music note

# Let's make the sound of the A5 note. (https://en.wikipedia.org/wiki/Piano_key_frequencies)

# In[27]:


# We want to have the note played for 0.5 seconds
duration = 0.5
# Define the rate
rate = DEFAULT_RATE
# The number of samples needed is
nsamples = int(rate*duration)
# # The frequency of A5 is 880. 
freq = 880
t = np.linspace(0, duration, nsamples)
data = np.sin(freq*2*np.pi*t)
#ipd.Audio(data,rate=rate)


# ### Write a function `make_note` 

# In[28]:


def make_note(freq, duration=0.3, rate=DEFAULT_RATE):
    '''
    receives as arguments:
        - frequency of the note (freq)
        - duration of the sound (set as default equal to 0.3)
        - rate (samples per second)
    and returns:
        - np.array data with the beep
    '''


# In[30]:


note_A5 = make_note(440)
# plt.plot(note_A5)
# ipd.Audio(note_A5,rate=DEFAULT_RATE)


# ### Modify the function `make_note`  so that it parabolically decays to zero over the time duration of the sound

# In[31]:


# We need a ramp function, which starts with value equal to 1 and finishes with value of zero,
# and includes nsamples data points
ramp = np.linspace(0, 1, nsamples)
# Here is the linear decay
# plt.plot(1-ramp)


# In[32]:


# Here is the parabolical decay
# plt.plot((1-ramp)**2)


# Modify the function `make_note` so that it applies the decay above to the data array

# Use your function to create the note A8 (freq=7040) with duration of 2 seconds

# In[34]:


data_A1 = make_note(7040,duration=2)
# plt.plot(data_A1)
# ipd.Audio(data_A1,rate=DEFAULT_RATE)


# ### Make music

# You can use numpy.hstack to combine notes to make music.
# Try to make a music by using the frequencies in `freq_example` consecutively, using the same duration for all notes. Store the combined array in the variable `music`.

# In[35]:


freq_example = [261.6256,293.6648,329.6276,349.2282,391.9954,440.0000,493.8833,523.2511]


# In[36]:


music = ...


# In[38]:


# ipd.Audio(music,rate=DEFAULT_RATE)


# What did you get?

# ### Name the music!

# note | count | freq
# --- | --- | --- 
# G | 0.5 | 392.0
# G | 0.5 | 392.0
# G | 0.5 | 392.0
# D | 0.5 | 293.66
# G | 0.5 | 392.0	
# B | 0.5 | 493.88
# A | 1 | 440.0
# B | 3 | 493.88

# ### Let's listen to movie sound clips

# In[43]:


# Name the movie!
# ipd.Audio("swnotry.wav")


# In[44]:


# Name the movie!
# ipd.Audio("honest.wav")


# ### Inspect the type of the data in `music_data`

# In[45]:


# scipy.io.wavfile.read: return the sample rate (in samples/sec) and data from a WAV file
filename = "honest.wav"
rate, music_data = wav.read(filename)
print("The sound has rate (in samples per second) = ", rate)
print("The data has", len(music_data), "sample points")
print(type(music_data))


# In[46]:


sound = np.array(music_data,dtype=float)
#plt.plot(sound)
# ipd.Audio(sound, rate=rate) 


# ### Change the speed of the sound

# Make it twice as fast

# Make it twice as slow

# ### Add noise to the sound

# Let's modify at random some of the elements of the numpy array. Use the function `array_add_noise` to create the variable `noisy_sound`

# In[59]:


# ipd.Audio(noisy_sound, rate=rate) 


# ### Scramble the sound!

# Check this one out! You can play with the number of splits.

# In[51]:


split_sound = np.array_split(sound, 8)
np.random.shuffle(split_sound)
flat_list = [item for sublist in split_sound for item in sublist]


# In[52]:


# ipd.Audio(np.array(flat_list), rate=rate) 


# ### Combine two different sounds

# In[53]:


filename1 = "odds.wav"
sr1, data1 = wav.read(filename1)
sound1 = np.array(data1,dtype=float)


# In[54]:


# ipd.Audio(data1,rate=sr1) 


# In[55]:


filename2 = "pass.wav"
sr2, data2 = wav.read(filename2)
sound2 = np.array(data2,dtype=float)


# In[56]:


# ipd.Audio(data2,rate=sr2)


# In[57]:


data_combined = np.hstack((0.01*sound1,sound2))
print(data1.shape, data2.shape,data_combined.shape)
# plt.plot(data_combined)


# In[58]:


# ipd.Audio(data_combined[100000:],rate=(sr1+sr2)/2 )


# We had to cheat and modify the magnitude of the first sound, so that we could hear both with similar volume. We also had to modify the rate, here using the average. Can you think of better ways to manipulate these two arrays to get something interesting? 
