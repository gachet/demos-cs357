
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# In[ ]:


s = 3 # seed
a = 37 #10 # multiplier
c = 2 # increment
m = 19 # modulus


# In[ ]:


n = 30
x = np.zeros(n)
x[0] = s
for i in range(1,n):
    x[i] = (a * x[i-1] + c) % m
#print(x)


# In[ ]:


plt.plot(x,'.')


# In[ ]:


s = 8
a = 1664525
c = 1013904223
m = 2**32


# In[ ]:


n = 30
x = np.zeros(n)
x[0] = s
for i in range(1,n):
    x[i] = (a * x[i-1] + c) % m
#print(x)


# In[ ]:


plt.plot(x,'.')

