#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


def func1(u, v, w):
    n = len(u)
    for i in range(n):
        w[i] = u[i] + 15.2 * v[i]
    
def func2(u, v, w):
    w = u + 15.2 * v


# In[6]:


n = 10**7
u = np.random.rand(n)
v = np.random.rand(n)
w = np.zeros(n)


get_ipython().run_line_magic('timeit', 'func1(u,v,w)')
get_ipython().run_line_magic('timeit', 'func2(u,v,w)')


# In[ ]:




