#!/usr/bin/env python
# coding: utf-8

# # Cost of Mat Mat multiply

# In[1]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt
get_ipython().run_line_magic('matplotlib', 'inline')
from time import process_time

def get_solve_time(n):
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    t_start = process_time()
    A @ B
    t_stop = process_time()
    
    return t_stop-t_start


# In[2]:


n_values = np.array([100,1000,2000,3000,4000,5000,6000,8000,10000])
print(n_values)


# In[3]:


times = []
for n in n_values:
    newtime = get_solve_time(n)
    times.append(newtime)


# In[4]:


pt.loglog(n_values, times)
pt.xlabel('n')
pt.ylabel('time')
pt.grid()


# * Can we predict individual values?
# * What does the overall behavior look like?
# * How could we determine the "underlying" function?

# In[ ]:




