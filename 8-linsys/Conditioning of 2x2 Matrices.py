
# coding: utf-8

# # Conditioning of $2\times2$ matrices
# 
# This mini-demo gives you the opportunity to play around with the 2-norm condition number of a $2\times 2$ matrix. 
# 
# * What happens if you choose the columns of the matrix to be nearly linearly dependent?
# * What happens if you choose the diagonal entries to be very different in magnitude?

# In[1]:


import numpy as np
import numpy.linalg as la


# In[3]:


la.cond([
         [0.000001, 0.1],
         [0,  1]
         ], 2)

