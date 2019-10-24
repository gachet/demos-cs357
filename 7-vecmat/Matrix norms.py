
# coding: utf-8

# # Matrix norms

# In[ ]:


import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# https://matplotlib.org/users/customizing.html
# print(plt.style.available) # uncomment to print all styles
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.figsize'] = (10.0, 8.0)


# Here's a matrix of which we're trying to compute the norm:

# In[ ]:


n = 2
A = np.random.randn(n, n)
A


# Recall:
# 
# $$||A||=\max_{\|x\|=1} \|Ax\|,$$
# 
# where the vector norm must be specified, and the value of the matrix norm $\|A\|$ depends on the choice of vector norm.
# 
# For instance, for the $p$-norms, we often write:
# 
# $$||A||_2=\max_{\|x\|=1} \|Ax\|_2,$$
# 
# and similarly for different values of $p$.

# --------------------
# We can approximate this by just producing very many random vectors and evaluating the formula:

# In[ ]:


xs = np.random.randn(n, 1000)
xs.shape


# First, we need to bring all those vectors to have norm 1. First, compute the norms:

# In[ ]:


p = 1
norm_xs = np.sum(np.abs(xs)**p, axis=0)**(1/p)
norm_xs.shape


# Then, divide by the norms and assign to `normalized_xs`:
# 
# Then check the norm of a randomly chosen vector.
# 
#  $${\rm normalized\_xs}= \frac{x}{||x||_p}$$

# In[ ]:


normalized_xs = xs/norm_xs
la.norm(normalized_xs[:, 99], p)


# Let's take a look at all `normalized_xs` vectors

# In[ ]:


plt.plot(normalized_xs[0], normalized_xs[1], "b.")
plt.gca().set_aspect("equal")


# Now apply $A$ to these normalized vectors:
# 
#  $${\rm A\_nxs}= A\frac{x}{||x||_p}$$

# In[ ]:


A_nxs = A.dot(normalized_xs)


# --------------
# Let's take a look again:

# In[ ]:


plt.plot(normalized_xs[0], normalized_xs[1], "b.", label="x")
plt.plot(A_nxs[0], A_nxs[1], "r.", label="Ax")
plt.legend()
plt.gca().set_aspect("equal")


# Next, compute norms of the $Ax$ vectors:
# 
#  $${\rm norm\_Axs}= ||Ax||_p$$

# In[ ]:


norm_Axs = np.sum(np.abs(A_nxs)**p, axis=0)**(1/p)
norm_Axs.shape


# What's the biggest one?

# In[ ]:


np.max(norm_Axs)


# Compare that with what `numpy` thinks the matrix norm is:

# In[ ]:


la.norm(A, p)

