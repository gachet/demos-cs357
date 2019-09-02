#!/usr/bin/env python
# coding: utf-8

# # Python Introduction: Types

# Let's evaluate some simple expressions.

# In[3]:


3*2


# In[4]:


5+3*2


# You can use `type()` to find the *type* of an expression.

# In[6]:


type(5.5+3*2)


# In[2]:


a = 5.5


# Now add decimal points.

# In[ ]:


5+3.5*2


# In[ ]:


type(5+3.0*2)


# Strings are written with single (``'``) or double quotes (`"`)

# In[8]:


'hello "hello"'

"hello 'hello'"


# Multiplication and addition work on strings, too.

# In[9]:


3 * 'hello ' + "cs357"


# Lists are written in brackets (`[]`) with commas (`,`).

# In[10]:


[5, 3.5, 7]


# In[11]:


type([5,3,7])


# List entries don't have to have the same type.

# In[12]:


["hi there", 15, [1,2,3]]


# "Multiplication" and "addition" work on lists, too.

# In[13]:


[1,2,3] * 4 + [5, 5, 5]


# Hmmmmmm. Was that what you expected?

# In[14]:


import numpy as np

np.array([1,2,3]) * 4 + np.array([5,5,5])


# In[ ]:




