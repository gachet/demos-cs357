
# coding: utf-8

# # Vector Norms

# ## Computing norms by hand
# 
# $p$-norms can be computed in two different ways in numpy:

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


# In[ ]:


x = np.array([1.,2,3])


# First, let's compute the 2-norm by hand:

# In[ ]:


np.sum(x**2)**(1/2)


# Next, let's use `numpy` machinery to compute it:

# In[ ]:


la.norm(x,2)


# Both of the values above represent the 2-norm: $\|x\|_2$.

# Different values of $p$ work similarly:

# In[ ]:


np.sum(np.abs(x)**5)**(1/5)


# In[ ]:


la.norm(x, 5)


# ## About the $\infty$-norm
# ---------------------
# 
# The $\infty$ norm represents a special case, because it's actually (in some sense) the *limit* of $p$-norms as $p\to\infty$.
# 
# Recall that: $\|x\|_\infty = \max(|x_1|, |x_2|, |x_3|)$.
# 
# Where does that come from? Let's try with $p=100$:

# In[ ]:


p=100
print(x)
print(x**p)


# In[ ]:


np.sum(x**p)


# Compare to last value in vector: the addition has essentially taken the maximum:

# In[ ]:


np.sum(x**p)**(1/p)


# Numpy can compute that, too:

# In[ ]:


la.norm(x, np.inf)


# -------------
# 
# ## Unit Balls
# 
# Once you know the set of vectors for which $\|x\|=1$, you know everything about the norm, because of semilinearity. The graphical version of this is called the 'unit ball'.
# 
# We'll make a bunch of vectors in 2D (for visualization) and then scale them so that $\|x\|=1$.

# In[ ]:


alpha = np.linspace(0, 2*np.pi, 200, endpoint=True)
x = np.cos(alpha)
y = np.sin(alpha)

vecs = np.array([x,y])

p = 1

norms = np.sum(np.abs(vecs)**p, axis=0)**(1/p)
norm_vecs = vecs/norms


plt.gca().set_aspect("equal")
plt.plot(norm_vecs[0], norm_vecs[1])
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

