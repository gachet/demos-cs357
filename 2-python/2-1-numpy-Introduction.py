#!/usr/bin/env python
# coding: utf-8

# # numpy: Introduction

# ## A Difference in Speed

# Let's import the `numpy` module.

# In[ ]:


import numpy as np


# In[ ]:


n = 5  # CHANGE ME
a1 = list(range(n)) # python list
a2 = np.arange(n)   # numpy array

if n <= 10:
    print(a1)
    print(a2)


# In[ ]:


get_ipython().run_line_magic('timeit', '[i**2 for i in a1]')


# In[ ]:


get_ipython().run_line_magic('timeit', 'a2**2')


# Numpy Arrays: much less flexible, but:
# 
# * much faster
# * less memory

# ## Ways to create a numpy array
# 
# * Casting from a list

# In[ ]:


a = np.array([1,2,3,5])
print(a)
print(a.dtype)


# In[ ]:


b = np.array([1.0,2.0,3.0])
print(b)
print(b.dtype)


# But also noticed that:

# In[ ]:


c = np.array([1,2,3])
print(c)
print(c.dtype)

d = np.array([1,2.,3])
print(d)
print(d.dtype)


# * `linspace`
# * np.linspace(start, stop, num=50,...)
# * num is the number of sample points

# In[ ]:


np.linspace(-1, 1, 9)


# * `zeros`

# In[ ]:


np.zeros((10,10), np.float64)


# Create 2D arrays, using zeros, using reshape and from list

# ## Operations on arrays
# 
# These propagate to all elements:

# In[ ]:


a = np.array([1.2, 3, 4])
b = np.array([0.5, 0, 1])


# Addition, multiplication, power ... are all elementwise:

# In[ ]:


a+b


# In[ ]:


a*b


# In[ ]:


a**b


# ## Important Attributes
# 
# Numpy arrays have two (most) important attributes:

# In[ ]:


A = np.random.rand(5, 4, 3)
A.shape


# The `.shape` attribute contains the dimensionality array as a tuple. So the tuple `(5,4,3)` means that we're dealing with a three-dimensional array of size $5 \times 4 \times 3$.
# 
# (`numpy.random.rand` just generates an array of random numbers of the given shape.)

# In[ ]:


A.dtype


# Other `dtype`s include `np.complex64`, `np.int32`, ...

# ## Iclicker question

# In[ ]:




#clear
A) c = [ 6 40 45]
B) c = [ 6 1000  225]
C) c = [ 6 30 30]
D) error message
# ## 1D arrays

# In[ ]:


a = np.random.rand(5)
a.shape


# In[ ]:


a = np.array([2,3,5])
print(a)
print(a.shape)


# ## 2D arrays

# In[ ]:


a = np.array([[2],[3],[5]])
print(a)
print(a.shape)

a = np.array([[2,3,5]])
print(a)
print(a.shape)


# We can change 1D numpy arrays into 2D numpy arrays using the function `reshape`

# In[ ]:


a = np.array([2,3,5]).reshape(3,1)
print(a)
print(a.shape)


# In[ ]:


a = np.array([2,3,5]).reshape(1,3)
print(a)
print(a.shape)


# In[ ]:


print(np.arange(1,10))
B = np.arange(1,10).reshape(3,3)
print(B)


# ## Transpose

# In[ ]:


print(B)


# In[ ]:


print(B.transpose())
print(B)


# In[ ]:


print(B.swapaxes(0,1))
print(B)


# In[ ]:


print(B.T)
print(B)


# In[ ]:


C = np.transpose(B)
print(C)


# What happens when we try to take the transpose of 1D array?

# In[ ]:


a = np.array([[2,3,5]])
print(a.T)


# But it works with 2D arrays

# In[ ]:


a = np.array([2,3,5]).reshape(3,1)
print(a)
print(a.T)


# ## Inner and outer products

# Matrix multiplication is `np.dot(A, B)` for two 2D arrays.

# In[ ]:


A = np.random.rand(3, 2)
B = np.random.rand(2, 4)
C = np.dot(A,B)
print(C.shape)

b = np.array([5,6])

d = np.dot(A,b)
print(d.shape)


# In[ ]:


A = np.array([[1,3],[2,4]])
B = np.array([[2,1],[3,2]])
print(np.dot(A,B))
print(A@B)


# In[ ]:


a = np.array([1,2,3])
b = np.array([5,6,7])
#Inner Product
print(np.dot(a,b))
print(np.inner(a,b))


# In[ ]:


#Outer Product C[i,j] = a[i]*b[j]
C = np.outer(a,b)
print(np.shape(C))
print(C)


# ## Iclicker question

# In[ ]:




#clear
 A) [[2 3]
     [2 6]]

 B) [[4 8]
     [7 9]]

 C) [[2 2]
     [3 6]]

 D) [[5 6]
     [10 8]]