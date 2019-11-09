
# coding: utf-8

# In[1]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# print(plt.style.available) # uncomment to print all styles
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12,6.0)
get_ipython().magic('matplotlib inline')


# In[2]:


def f(x):
    return 0.8 - x + x**2


# In[3]:


xx = np.linspace(0,1,1000)
plt.plot(xx, f(xx))


# ## Add noise for some random samples

# In[16]:


n = 25
t = np.random.rand(n)
y = f(t) + 0.1*np.random.randn(n)*f(t)


# In[17]:


plt.plot(xx, f(xx))
plt.plot(t, y, 'ro')


# ### Find a quadratic fit

# In[18]:


A = np.zeros((n, 3))
A[:,0] = 1
A[:,1] = t
A[:,2] = t**2


# In[19]:


ATA = np.dot(A.T, A)
ATb = np.dot(A.T, y)
x = la.solve(ATA, ATb)
print(x)


# In[20]:


x0,x1,x2 = x

plt.plot(xx, f(xx))
plt.plot(t, y, 'ro')
plt.plot(xx, x0 + x1*xx + x2*xx**2, 'g-', lw=3)

