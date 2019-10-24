
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from math import factorial

import sympy as sp
sp.init_printing()


# In[ ]:


def plot_sympy(my_f, my_pts, **kwargs):
    f_values = np.array([my_f.subs(x, pt) for pt in my_pts])
    plt.plot(pts, f_values, **kwargs)


# In[ ]:


sp.var("x")
x


# ## Polynomial Approximation

# In[ ]:


f = sp.sqrt(1-x**2)


# In[ ]:


f


# In[ ]:


print(f.diff(x,1).subs(x, 0))
print(f.diff(x,2).subs(x, 0))
print(f.diff(x,3).subs(x, 0))
print(f.diff(x,4).subs(x, 0))
print(f.diff(x,5).subs(x, 0))


# In[ ]:


n = 2


# In[ ]:


tn = 0
for i in range(n+1):
    tn += f.diff(x, i).subs(x, 0)/factorial(i) * x**i


# In[ ]:


tn


# In[ ]:


plot_sympy(tn, pts, label="taylor")
plot_sympy(f, pts, label="f")
plt.legend(loc="best")
plt.ylim([-1.3, 1.3])
plt.axis('equal')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('function values')


# ## Behavior of the Error

# In[ ]:


plot_sympy(error, pts, label="error")
plt.legend(loc="best")
plt.ylim([-1.3, 1.3])
plt.axis('equal')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('error')


# To get a better idea of what happens close to the center, use a log-log plot:

# In[ ]:


plt.figure(figsize=(10,6))
# plot only points close to zero [10^(-3),10^(0.5)]
pos_pts = 10**np.linspace(-3, 0.5) 
err_values = [abs(error.subs(x, pt)) for pt in pos_pts]
plt.loglog(pos_pts, err_values)
plt.grid()
plt.xlabel("$x$")
plt.ylabel("Error")


# What is the slope of the error plot? 
