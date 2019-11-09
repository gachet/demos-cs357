
# coding: utf-8

# # Computing the SVD

# In[27]:


import numpy as np
import numpy.linalg as la


# ## 1) For a square matrix

# In[37]:


m = 4
n = m
A = np.random.randn(m, n)
print(A)


# ### Using numpy.linalg.svd

# In[60]:


U, S, Vt = la.svd(A)


# In[61]:


print(U)
print(U.shape)


# In[62]:


print(Vt)
print(Vt.T.shape)


# In[63]:


print(S)
print(S.shape)


# ### Using eigen-decomposition

# Now compute the eigenvalues and eigenvectors of $A^TA$ as `eigvals` and `eigvecs`

# In[64]:


eigvals, eigvecs = la.eig(A.T.dot(A))


# Eigenvalues are real and positive. Coincidence?

# In[65]:


eigvals


# `eigvecs` are orthonormal! Check:

# In[66]:


eigvecs.T @ eigvecs 


# Now piece together the SVD:

# In[73]:


S2 = np.sqrt(eigvals)
V2 = eigvecs
U2 = A @ V2 @ la.inv(np.diag(S2))


# ## 2) For a nnon-square square matrix

# In[89]:


m = 3
n = 5
A = np.random.randn(m, n)
print(A)


# In[90]:


U, S, Vt = la.svd(A,full_matrices=True)


# In[91]:


print(U)
print(U.shape)

print(Vt)
print(Vt.T.shape)

print(S)
print(S.shape)


# In[92]:


eigvals, eigvecs = la.eig(A.T.dot(A))
print(eigvals)

