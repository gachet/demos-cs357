
# coding: utf-8

# # Polynomial Approximation with Derivatives

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from math import factorial


# ## A Brief Intro to `sympy`

# Here we import `sympy`, a package for symbolic computation with Python.

# In[ ]:


import sympy as sp
sp.init_printing()


# Next, we make a (symbolic) variable $x$ from which we can then build more complicated expressions:

# In[ ]:


sp.var("x")
x


# Build up an expression with $x$. Assign it to `expr`. Observe that this expression isn't evaluated--the result of this computation is some Python data that represents the computation:

# In[ ]:


g = sp.sin(sp.sqrt(x)+2)**2
g


# Next, take a derivative, using `.diff(x)`.

# In[ ]:


g.diff(x)


# In[ ]:


g.diff(x, 4)


# Use `.subs(x, ...)` and `.evalf()` to evaluate your expression for $x=1$.

# In[ ]:


g.subs(x,1)


# In[ ]:


g.subs(x, 1).evalf()


# Helper function that takes symbolic functions as argument and plot them

# In[ ]:


def plot_sympy(my_f, my_pts, **kwargs):
    f_values = np.array([my_f.subs(x, pt) for pt in my_pts])
    plt.plot(pts, f_values, **kwargs)


# ## Polynomial Approximation

# In[ ]:


f = 1/(20*x-10)
f


# In[ ]:


f.diff(x)


# In[ ]:


f.diff(x,2)


# Write out the degree 2 Taylor polynomial about 0:

# In[ ]:


taylor2 = (
    f.subs(x, 0)
    + f.diff(x).subs(x, 0) * x
    + f.diff(x, 2).subs(x, 0)/2 * x**2
)
taylor2


# Plot the exact function `f` and the taylor approximation `taylor2`

# In[ ]:


pts = np.linspace(-0.4,0.4)

plot_sympy(taylor2, pts, label="taylor2")
plot_sympy(f, pts, label="f")
plt.legend(loc="best")
plt.axis('equal')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('function values')


# ## Behavior of the Error

# Let's write the taylor approximation for any degree `n`, and define the error as f - tn

# In[ ]:


n = 2

tn = 0
for i in range(n+1):
    tn += f.diff(x, i).subs(x, 0)/factorial(i) * x**i


# In[ ]:


plot_sympy(tn, pts, label="taylor")
plot_sympy(f, pts, label="f")
plt.legend(loc="best")
plt.axis('equal')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('function values')


# In[ ]:


error = f - tn


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


# plot only points close to zero [10^(-3),10^(0.4)]
plt.figure(figsize=(10,6))
pos_pts = 10**np.linspace(-3, 0.4) 
err_values = [abs(error.subs(x, pt)) for pt in pos_pts]
plt.loglog(pos_pts, err_values)
plt.grid()
plt.xlabel("$x$")
plt.ylabel("Error")

