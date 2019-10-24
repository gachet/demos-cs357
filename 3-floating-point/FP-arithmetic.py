
# coding: utf-8

# In[ ]:


import numpy as np


# Distributive property

# In[ ]:


print(100*(0.1 + 0.2))
print(100*0.1 + 100*0.2)


# In[ ]:


100*(0.1 + 0.2) == 100*0.1 + 100*0.2


# Associative property

# In[ ]:


(np.pi+1e100)-1e100


# In[ ]:


(np.pi)+(1e100-1e100)


# In[ ]:


b = 1e80
a = 1e2
print(a + (b - b) )
print((a + b) - b )

