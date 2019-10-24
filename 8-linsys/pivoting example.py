
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.linalg as sla



# # Let's try an example
# 

# In[ ]:


A = np.array([
    [2,1,1,0],
    [4,3,3,1],
    [8,7,9,5],
    [6,7,9,8]
])
np.set_printoptions(precision=2, suppress=True)
print(A)


# In[ ]:


P, L, U = sla.lu(A)


# In[ ]:


print(P.T)
print(L)
print(U)


# In[ ]:


print(np.dot(P.T, A))


# In[ ]:


print(np.dot(L, U))

