
# coding: utf-8

# # Newton's Method

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set(font_scale=2)
sns.set_style("whitegrid")


# Here's a function:

# In[2]:


def f(x):
    return x**3 - x +1

def df(x):
    return 3*x**2 - 1

xmesh = np.linspace(-4, 4, 100)
plt.ylim([-5, 10])
plt.plot(xmesh, f(xmesh))


# In[8]:


guesses = [-.9]
guesses = [1.5]


# Evaluate this cell many times in-place (using Ctrl-Enter)

# In[29]:


x = guesses[-1] # grab last guess

slope = df(x)

# plot approximate function
plt.plot(xmesh, f(xmesh))
plt.plot(xmesh, f(x) + slope*(xmesh-x))
plt.plot(x, f(x), "o")
plt.xlim([-4, 4])
plt.ylim([-5, 10])
plt.axhline(0, color="black")

# Compute approximate root
xnew = x - f(x) / slope
guesses.append(xnew)
print(xnew)


# In[16]:


f(xnew)


# In[ ]:


print(guesses)

