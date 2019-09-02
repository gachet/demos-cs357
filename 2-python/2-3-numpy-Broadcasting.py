#!/usr/bin/env python
# coding: utf-8

# # numpy: Broadcasting

# In[ ]:


import numpy as np


# In[ ]:


a = np.arange(9).reshape(3,3)
print(a.shape)
print(a)


# In[ ]:


b = np.arange(4, 4+9).reshape(3, 3)
print(b.shape)
print(b)


# In[ ]:


a+b


# So this is easy and one-to-one.
# 

# ---
# 
# What if the shapes do not match?

# In[ ]:


a = np.arange(9).reshape(3, 3)
print(a.shape)
print(a)


# In[ ]:


b = np.arange(3)
print(b.shape)
print(b)


# What will this do?

# In[ ]:


a+b


# It has *broadcast* along the last axis!

# ---
# 
# Can we broadcast along the *first* axis?

# In[ ]:


a


# In[ ]:


c = b.reshape(3, 1)
c


# In[ ]:


print(a.shape)
print(c.shape)


# In[ ]:


a+c


# Rules:
# 
# * Shapes are matched axis-by-axis from last to first.
# * A length-1 axis can be *broadcast* if necessary.

# ## Iclicker question

# In[ ]:





# In[ ]:




#clear
Mark the correct output for print(B + np.dot(A,c))

# A) [[3 5]
#     [6  8]]

# B) [[ 3 11]
#     [ 3  8]]

# C) [[11 10]
#     [12  9]]

# D) [[11 13]
#     [9  9]]

# In[ ]:




