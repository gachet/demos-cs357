
# coding: utf-8

# # numpy: Indexing

# In[1]:


import numpy as np


# In[2]:


A = np.array([[1, 4, 9], [2, 8, 18]])
print(A)


# In[3]:


A[1,2]


# What's the result of this?

# In[4]:


A[:,1]


# And this?

# In[5]:


A[1:,:1]


# One more:

# In[ ]:


A[:,[0,2]]


# ## Iclicker questions
#clear
Select the correct output:

A) <class 'int'>

B) <class 'float'>

C) <class 'numpy.int64'>

D) <class 'numpy.float64'>#clear
Select the correct output:

 A) [[ 8  3 18]
     [ 5  1  2]]

 B) [[ 3 18]
     [ 1  2]]    

 C) [[3 18]]

 D) [[8 3 18]]
# For higher-dimensional arrays we can use `...` like:

# In[ ]:


a = np.random.rand(3,4,2)
a.shape


# In[ ]:


a[...,1].shape


# ---
# 
# Indexing into numpy arrays usually results in a so-called *view*.

# In[ ]:


a = np.zeros((4,4))


# Let's call `b` the top-left $2\times 2$ submatrix.

# In[ ]:


b = a[:2,:2]


# What happens if we change `b`?

# In[ ]:


b[1,0] = 5


# In[ ]:


a


# To decouple `b` from `a`, use `.copy()`.

# In[ ]:


b = b.copy()
b[1,1] = 7
print(b)
print(a)


# ## iclicker question
#clear
What is the output of print(A)?

 A) [[ 1  4  9  3]
     [ 2  8  3 18]
     [ 4  5  2  2]
     [ 6  4  6  3]]

 B) [[ 1  4  9  3]
     [ 2  8  3 18]
     [ 4  5  1  2]
     [ 6  4  6  3]]
     
 C) none of the above
# ---
# 
# You can also index with boolean arrays:

# In[ ]:


a = np.random.rand(4,4)


# In[ ]:


a


# In[ ]:


a_big = a>0.5
a_big


# In[ ]:


a[a_big]


# Also each index individually:

# In[ ]:


a_row_sel = [True, True, False, True]


# In[ ]:


a[a_row_sel,:]


# ---
# 
# And with index arrays:

# In[ ]:


a


# In[ ]:


x,y = np.nonzero(a > 0.5)


# In[ ]:


x


# In[ ]:


y


# In[ ]:


a[(x,y)]

