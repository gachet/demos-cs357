
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

params = {'legend.fontsize': 20,
          'figure.figsize': (12, 6),
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20,
         'lines.linewidth': 3}
plt.rcParams.update(params)


# #  Make some data
# 
# Let's make 3 pieces of data:
# 
# * `error1` might represent error that looks like $e\sim n^{-1}$
# * `error15` might represent error that looks like $e\sim n^{-1.5}$
# * `error2` might represent error that looks like $e\sim n^{-2}$
# 
# ### What should this look like in linear, log-linear, and log-log?
# 
# This is called *algebraic* convergence, with an algebraic index of convergence of $\alpha = 1.0, 1.5, 2.0$, where
# $$
# e \sim n^{-\alpha}
# $$

# In[4]:


n = np.logspace(1, 6, 100) # evenly distribute numbers over logspace

error1 = 1 / n**1
error15 = 1 / n**1.5
error2 = 1 / n**2

p = plt.plot
p(n, error1, label=r'$n^{-1}$')
p(n, error15, label=r'$n^{-1.5}$')
p(n, error2, label=r'$n^{-2}$')

plt.xlabel('$n$')
plt.ylabel('error')
plt.grid()
plt.legend(frameon=False)


# # Think about *faster* convergence than algebraic
# 
# Let's make 3 pieces of data:
# 
# * `error21` might represent error that looks like $e\sim 2^{-n}$
# * `error23` might represent error that looks like $e\sim 2^{-3n}$
# * `error2e` might represent error that looks like $e\sim e^{-2n}$
# 
# ### What should this look like?
# 
# Here the algebraic index is unbounded (the error decays fastter than $n^{-\alpha}$ for any $\alpha$).   So we call this **exponential** or **spectral** or a form of **geometric** convergence.
# 
# That is
# $$
# e \sim e^{-\mu n}
# $$
# for some rate $\mu$ of exponential convergence.

# In[7]:


n = np.logspace(0, 1, 100) # evenly distribute numbers over logspace

error21 = 2**-n
error23 = 2**-(3*n)
error2e  = np.exp(-2*n)

p = plt.plot
p(n, error21, label=r'$2^{-n}$')
p(n, error23, label=r'$2^{-3n}$')
p(n, error2e, label=r'$e^{-2n}$')

plt.xlabel('$n$')
plt.ylabel('error')
plt.grid()
plt.legend(frameon=False)


# In[11]:


n = np.logspace(0, 2, 100) # evenly distribute numbers over logspace

error2e  = np.exp(-2*n)
error2 = 1 / n**2

p = plt.plot

p(n, error2e, label=r'$e^{-2n}$')
p(n, error2, label=r'$n^{-2}$')

plt.xlabel('$n$')
plt.ylabel('error')
plt.grid()
plt.legend(frameon=False)

