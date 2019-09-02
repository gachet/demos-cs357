#!/usr/bin/env python
# coding: utf-8

# # numpy: Tools

# ### Other tools
# 
# * `numpy.linalg`
# * `scipy`
# * `matplotlib`

# In[ ]:


import numpy as np
import matplotlib.pyplot as pt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


x = np.linspace(-10, 10, 100)
pt.plot(x, np.sin(x)+x*0.5)


# In[ ]:


pic = np.sin(x).reshape(len(x), 1)  * np.cos(x)
pt.imshow(pic)

