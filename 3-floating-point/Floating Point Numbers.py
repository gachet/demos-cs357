
# coding: utf-8

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np


# # Floating Point Numbers

# ### Integer Range

# We discussed in class that we can obtain the range of integers that can be represented exactly as
# 
# $$[1,2^{p}]$$
# 
# So let's try this using Python (where double precision gives p=53)

# In[ ]:


# This number is within the range
2**52


# In[ ]:


# We add one to get the next integer 
2**52 + 1


# In[ ]:


# All good so far, right? Let's check the integer:
2**53


# In[ ]:


# If I add one, what do you expect us to get?
2**53 + 1


# Humm... not what you expected right? Why do you think this happened?

# ### Integer type

# In[ ]:


# to find the maximum integer (using 64-bit signed integer)
# uses only 63 bits, since one is reserved for the sign
maxint = 0
for i in range(63):
     maxint += 2**i
maxint


# ### What will happen when we run these code snippets?

# A) it won't stop (infinite loop)
# 
# B) it will stop when b reaches underflow
# 
# C) it will stop when b reaches machine epsilon
# 
# D) none of the above

# In[ ]:


a = 10**4
b = 1.0
i=0
while (a + b > a):
    #print("{0:d}, {1:1.16f}". format(i, b)) 
    print(i,b)
    b = b / 2.0
    i+=1


# A) it won't stop (infinite loop)
# 
# B) it will stop when b reaches underflow
# 
# C) it will stop when b reaches machine epsilon
# 
# D) none of the above

# In[ ]:


a = 1.0
while a > 0.0: 
    a = a / 2.0
    print(a)
    #print("% .16e"% (a)) 


# In[ ]:


a = 1.0
while a != np.inf:
    a = a * 2.0
    print(a)
    #print("% .16e"% (a)) 


# Let's write the following number (not exacly OFL, but close)

# In[ ]:


float(2**1023)


# In[ ]:


# Try the UFL definition
2**(1024)*(1-2**(-53))

# Can we make this work?

