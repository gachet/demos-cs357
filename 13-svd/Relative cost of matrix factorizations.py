
# coding: utf-8

# # Relative cost of matrix factorizations

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
from time import time


# In[2]:


# https://matplotlib.org/users/customizing.html
# print(plt.style.available) # uncomment to print all styles
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.figsize'] = (10.0, 8.0)


# In[3]:


n_values = (10**np.linspace(1, 3, 15)).astype(np.int32)
n_values


# In[6]:


def matmat(A):
    A @ A

for name, f in [
        ("lu", spla.lu_factor),
        ("matmat", matmat),
        ("svd", npla.svd)
        ]:

    times = []
    print("----->", name)
    
    for n in n_values:
        A = np.random.randn(n, n)
        
        start_time = time()
        f(A)
        delta_time = time() - start_time
        times.append(delta_time)
        
        print("%d - %f" % (n, delta_time))
        
    plt.plot(n_values, times, label=name)

plt.legend(loc="best")
plt.xlabel("Matrix size $n$")
plt.ylabel("Wall time [s]");

