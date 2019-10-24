
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.linalg as sla
import scipy.sparse as sparse

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Topology design optimization

# ![](myfigure.png)

# When performing optimization of structural problem, for example to obtain the bridge design above, you may want to use a numerical method called Finite Element Method. The optimization will consist of a series of `solve` of the form:
# $$ {\bf K} {\bf u} = {\bf F} $$
# 
# Here will load the matrix $ {\bf K}$ from a file. The matrix is given in Compressed Sparse Column (CSC) format.

# In[ ]:


K = sparse.load_npz('yourmatrix.npz')
K


# In[ ]:


K.shape[0]**2/105912


# We can `spy` the distribution of the non-zero entries of the matrix:

# In[ ]:


plt.spy(K)
plt.show()


# The matrix ${\bf K}$ has a banded format, and it is also symmetric and positive definite. 

# In[ ]:


Kdense = K.todense()


# In[ ]:


np.max(Kdense-Kdense.T)


# In[ ]:


sla.norm(Kdense-Kdense.T)


# Solving the linear system of equations using different methods:

# In[ ]:


F = np.zeros(K.shape[0])
F[1]=-1


# ### LU factorization

# In[ ]:


u1 = sla.solve(Kdense,F)
u1.shape


# In[ ]:


get_ipython().magic('timeit sla.solve(Kdense,F)')


# In[ ]:


lu,p = sla.lu_factor(Kdense)
u2 = sla.lu_solve((lu,p),F)
u2.shape


# In[ ]:


get_ipython().magic('timeit sla.lu_factor(Kdense)')
get_ipython().magic('timeit sla.lu_solve((lu,p),F)')


# ### Cholesky factorization

# In[ ]:


Kcho = sla.cholesky(Kdense)
u3 = sla.cho_solve((Kcho,False),F)
u3.shape


# In[ ]:


get_ipython().magic('timeit sla.cholesky(Kdense)')
get_ipython().magic('timeit sla.cho_solve((Kcho,False),F)')


# ### Sparse solve

# In[ ]:


from scipy.sparse.linalg import spsolve


# In[ ]:


u4 = spsolve(K,F)
u4.shape


# In[ ]:


get_ipython().magic('timeit spsolve(K,F)')

