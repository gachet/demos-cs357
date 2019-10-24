
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# In[2]:


np.random.seed(908)  # a not-so-random starting seed


# Let's start with a uniform distribution on the unit square  [0,1]×[0,1] . Create a 2D array samples of shape (2, nsamples):

# In[10]:


nsamples = 10**2
samples = np.random.rand(2, nsamples)


# Scale the sample points "samples", so that we have a uniform distribution inside a square $[-1,1]\times [-1,1]$. Calculate the distance from each sample point to the origin $(0,0)$

# In[11]:


xy = samples * 2 - 1.0 # scale sample points
r = np.sqrt(xy[0, :]**2 + xy[1, :]**2)  # calculate radius

## talk about here how we can reduce the calculation by removing the sqrt
plt.plot(xy[0,:], xy[1,:], 'k.')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y');


# If the distance is less than 1 (radius of the circle), then the sample point is inside the circle.

# In[12]:


incircle = (r <= 1.0)
n = np.arange(1, nsamples+1)
countincircle = 4 * incircle.cumsum() / n


# The approximation for $\pi$ when we use `n` sample points is:

# In[13]:


pi_approx = countincircle[-1]
print(pi_approx)


# In[14]:


plt.plot(xy[0,np.where(incircle)[0]], xy[1,np.where(incircle)[0]], 'b.')
plt.plot(xy[0,np.where(incircle==False)[0]], xy[1,np.where(incircle==False)[0]], 'r.')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y');


# ## Aproximation for $\pi$ for increasing values of `n` (number of sample points)

# In[15]:


plt.plot(n, countincircle, '.')
plt.xlabel('n')
plt.ylabel('count in circle');


# In[17]:


error = np.abs(countincircle - np.pi)
plt.loglog(n, error, '.')
plt.xlabel('n')
plt.ylabel('error')

