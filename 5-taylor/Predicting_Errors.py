
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from math import factorial

import sympy as sp
sp.init_printing()


# ## Predicting Taylor Error

# In[ ]:


sp.var("x")
x


# In[ ]:


f = sp.sqrt(x-10)


# In[ ]:


n = 3
x0 = 12

tn = 0
for i in range(n+1):
    tn += f.diff(x, i).subs(x, x0)/factorial(i) * (x-x0)**i
tn


# The error of the Taylor approximation of degree 3 about x0 = 12 when h=0.5 is (that is, x = 12.5):

# In[ ]:


f.subs(x, 12.5)


# In[ ]:


t.subs(x, 12.5).evalf()


# In[ ]:


error1 = f.subs(x, 12.5) - t.subs(x, 12.5).evalf()
abs(error1)


# Now predict the error at $12.25$:

# and the actual error is:

# In[ ]:


error2 = f.subs(x, 12.25) - t.subs(x, 12.25).evalf()
abs(error2)

