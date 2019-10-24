
# coding: utf-8

# In[1]:


import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


# In[2]:


def lu1(A):
    """
    """
    LU = A.copy()
    
    n = A.shape[0]
    for i in range(1,n):
        l_21 = LU[i:,i-1]
        u_12 = LU[i-1,i:]
        A_22 = LU[i:,i:]
        u_11 = LU[i-1,i-1]
        
        # l_{21} = a_{21} / u_{11}
        l_21 /= u_11
        # A_{22} = LU[] 
        A_22 += -np.outer(l_21, u_12)
        
    return LU


# # Try this
# $$
# Ax = \begin{bmatrix}c&1\\-1&1\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}
# =
# \begin{bmatrix}b_1\\b_2\end{bmatrix}
# $$
# with an exact solution of
# $$
# x_{exact} = \begin{bmatrix}1\\1\end{bmatrix}
# $$

# In[10]:


c = 1e-1
A = np.array([[c, 1.], [-1, 1]])
# xx is the exact solution
xx = np.array([1,1])
b = A.dot(xx)

# Comput the LU
LU = lu1(A)
L = np.tril(LU,-1) + np.eye(2,2)
U = np.triu(LU)

# Solve
# x is the numerical (xhat)
y = sla.solve_triangular(L, b, lower=True)
x = sla.solve_triangular(U, y)


print("Condition number = ", la.cond(A,2))

print("Exact solution = ", xx)

print("Computed solution = ",x)

print("Error = ", la.norm(xx-x))


# ## Iclicker

# In[ ]:


#Is the matrix A singular
# A) YES
# B) NO


# In[9]:


la.inv(A)


# ## Residual

# In[ ]:


c = 1e-1
A = np.array([[c, 1.], [-1, 1]])
xx = np.array([1,1])
b = A.dot(xx)

# Comput the LU
LU = lu1(A)
L = np.tril(LU,-1) + np.eye(2,2)
U = np.triu(LU)

# Solve
y = sla.solve_triangular(L, b, lower=True)
x = sla.solve_triangular(U, y)


print("Exact solution = ", xx)

print("Computed solution = ",x)

print("Condition number = ", la.cond(A,2))

print("Residual norm = ",la.norm(A@x - b))

print("Error norm = ",la.norm(xx - x))

