
# coding: utf-8

# # Relative cost of matrix operations

# In[1]:


import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as sla
get_ipython().magic('matplotlib inline')


from time import time


# In[2]:


n_values = (10**np.linspace(1, 4, 15)).astype(np.int32)
n_values


# In[3]:


times_matmul = []
times_lu = []

for n in n_values:
    print(n)
    A = np.random.randn(n, n)
    start_time = time()
    A.dot(A)
    times_matmul.append(time() - start_time)
    start_time = time()
    la.lu(A)
    times_lu.append(time() - start_time)


# In[4]:


plt.plot(n_values, times_matmul, label='matmul')
plt.plot(n_values, times_lu, label='lu')
plt.grid()
plt.legend(loc="best")
plt.xlabel("Matrix size $n$")
plt.ylabel("Wall time [s]")


# * The faster algorithms make the slower ones look bad. But... it's all relative.
# * Can we get a better plot?
# * Can we see the asymptotic cost ($O(n^3)$) of these algorithms from the plot?

# In[ ]:


plt.loglog(n_values, times_matmul, label='matmul')
plt.loglog(n_values, times_lu, label='lu')
plt.grid()
plt.legend(loc="best")
plt.xlabel("Matrix size $n$")
plt.ylabel("Wall time [s]")

