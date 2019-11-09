
# coding: utf-8

# # Data Fitting with Least Squares

# In[1]:


import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Suppose we are modeling a relationship between $x$ and $y$, and the "true" relationship is $y = a+bx$:

# In[2]:


a = 4
b = 2

def f(pts):
    return a + b*pts

plot_grid = np.linspace(-3, 3, 100)
plt.plot(plot_grid, f(plot_grid))


# But suppose we don't know $a$ and $b$, but instead all we have is a few noisy measurements (i.e. here with random numbers added):

# In[3]:


npts = 5

#np.random.seed(22)
points = np.linspace(-2, 2, npts) + np.random.randn(npts)
values = f(points) + .5*np.random.randn(npts)*f(points)

plt.plot(plot_grid, f(plot_grid))
plt.plot(points, values, "or")


# What's the system of equations for $a$ and $b$? We will solve the least squares problem by solving the Normal Equations $$ A^T A x = A^T b$$

# What's the right-hand side vector?

# In[5]:


Atb = A.T@values


# In[6]:


x = la.solve(AtA,Atb)


# In[7]:


x


# Recover the computed $a$, $b$:

# In[8]:


a_c, b_c = x


# In[9]:


def f_c(pts):
    return a_c + b_c * pts

plt.plot(plot_grid, f(plot_grid), label="true", color="green")
plt.plot(points, values, "o", label="data", color="blue")
plt.plot(plot_grid, f_c(plot_grid), "--", label="best fit",color="purple",)

plt.legend()

