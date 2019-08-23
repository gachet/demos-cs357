
# coding: utf-8

# In[ ]:


from time import process_time
import numpy as np
from time import process_time


# In[ ]:


n = 2000
A = np.random.randn(n,n)
B = np.random.randn(n,n)

t = process_time()  # store the time
C = A @ B
t = process_time() - t
print(t)


# In[ ]:


n = 2000
a = 2
A = np.random.randn(a*n,a*n)
B = np.random.randn(a*n,a*n)

t2 = process_time()  # store the time
C = A @ B
t2 = process_time() - t2
print(t2)


# In[ ]:


t2/t

