
# coding: utf-8

# # Playing around with Random Number Generators

# In[ ]:


import numpy as np


# In[ ]:


np.random.rand(10)


# In[ ]:


for x in range(0, 10):
    numbers = np.random.rand(10)
    print(numbers)


# In[ ]:


np.random.seed(10)

