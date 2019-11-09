
# coding: utf-8

# # Data Fitting with Least Squares

# In[1]:


import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import random
import matplotlib.pyplot as pt
get_ipython().magic('matplotlib inline')


# The function below creates a random matrix and a random right-hand side vector, to use as input data for least squares. The arguments of the function are the shape of A, and the rank of A. You should run examples to investigate the following situations:
# 
# 1) rankA = N (this is a full rank matrix, and hence solution is unique
# 
# 2) rankA = N - 1 (this is a rank deficient matrix, and the solution is no longer unique

# In[2]:


def creates_A_b(shap = (10,4), rankA=4):
    M,N = shap
    # Generating the orthogonal matrix U
    X = np.random.rand(M,M)
    U,R = sla.qr(X)
    # Generating the orthogonal matrix V
    Y = np.random.rand(N,N)
    V,R = sla.qr(Y)
    Vt = V.T
    # Generating the diagonal matrix Sigma
    singval = random.sample(range(1, 9), rankA)
    singval.sort()
    sigmavec = singval[::-1]
    sigma = np.zeros((M,N))
    for i,sing in enumerate(sigmavec):
        sigma[i,i] = sing
    A = U@sigma@Vt
    b = np.random.rand(M)
    return(A,b)


# In[3]:


# Matrix shape
M = 10
N = 4
A,b = creates_A_b((M,N),N)


# In[4]:


print(la.cond(A))


# ## Using normal equations (unique solution, full rank)

# In[5]:


xu = la.solve(A.T@A,A.T@b)
print(xu)


# In[6]:


la.norm(A@xu-b,2)


# In[7]:


la.norm(xu,2)


# ## Using SVD

# In[8]:


UR,SR,VRt = la.svd(A,full_matrices=False)
print(SR)


# In[9]:


ub = (UR.T@b)
x = np.zeros(N)
for i,s in enumerate(SR):
    if s > 1e-15:
        x += VRt[i,:]*ub[i]/s
print(x)


# In[10]:


la.norm(A@x-b,2)


# In[11]:


la.norm(x,2)


# ## Using numpy.linalg Least Squares method

# In[12]:


coeffs,residual,rank,sval=np.linalg.lstsq(A,b,rcond=None)
print(coeffs)


# In[13]:


la.norm(A@coeffs-b,2)


# In[14]:


la.norm(coeffs,2)


# In[15]:


residual


# In[16]:


rank

