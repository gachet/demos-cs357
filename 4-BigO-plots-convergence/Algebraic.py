
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# ## Power functions:

# In[ ]:


a = 4
b = -2
x = np.arange(1,1e5)
y = a*x**(b)


# In[ ]:


plt.plot(x,y,'-',lw=3)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)


# ## Exponential function

# In[ ]:


a = 4.0
b = -1.0
x = np.arange(1,100)
y = a**(b*x)


# In[ ]:


plt.plot(x,y,'-', lw=3)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)


# ## Log functions

# In[ ]:


a = 4.0
b = 1.0
x = np.arange(1,100)
y = a*np.log(b*x)


# In[ ]:


plt.plot(x,y,'-', lw=3)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

