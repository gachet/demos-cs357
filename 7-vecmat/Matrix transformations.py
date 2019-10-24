
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (6, 6.0)


# # Make a grid of data
# 
# What does this do?  It makes `nx` points on each x-line and `ny` points on each y-line.

# In[2]:


nx = 100
ny = 100
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)

gridlines = np.linspace(0,1,8)
data = []
for g in gridlines:
    data.append(np.vstack((x,0*y+g)))
    data.append(np.vstack((0*x+g,y)))
data = np.hstack(list(data))
print(data.shape)


# In[3]:


plt.plot(data[0,:], data[1,:], 'go')


# ## Try some transformations
# 
# https://en.wikipedia.org/wiki/Matrix_(mathematics)#Linear_transformations
# 
# Shear:
# $$
# \begin{bmatrix}
# 1 & 1.25  \\
# 0 & 1
# \end{bmatrix}
# \begin{bmatrix}
# x\\y
# \end{bmatrix}
# $$
# 
# Reflection:
# $$
# \begin{bmatrix}
# -1 & 0  \\
# 0 & 1
# \end{bmatrix}
# \begin{bmatrix}
# x\\y
# \end{bmatrix}
# $$
# 
# Shrink:
# $$
# \begin{bmatrix}
# 3/2 & 0  \\
# 0 & 2/3
# \end{bmatrix}
# \begin{bmatrix}
# x\\y
# \end{bmatrix}
# $$
# 
# Expand:
# $$
# \begin{bmatrix}
# 3/2 & 0  \\
# 0 & 3/2
# \end{bmatrix}
# \begin{bmatrix}
# x\\y
# \end{bmatrix}
# $$
# 
# Rotate:
# $$
# \begin{bmatrix}
# \cos(\pi / 6) & -\sin(\pi / 6)\\
# \sin(\pi / 6) & \cos(\pi / 6)
# \end{bmatrix}
# \begin{bmatrix}
# x\\y
# \end{bmatrix}
# $$
# 
# Move:
# $$
# \begin{bmatrix}
# 1 & 0\\
# 0 & 1\\
# \end{bmatrix}
# \begin{bmatrix}
# x\\y
# \end{bmatrix}
# + 
# \begin{bmatrix}
# xshift\\
# yshift
# \end{bmatrix}
# $$

# #### Shear

# In[9]:


A = np.array([[2,0],[0,1]])
print(A)

newdata = A.dot(data)
plt.plot(newdata[0,:], newdata[1,:], 'bo')
plt.plot(data[0,:], data[1,:], 'go')
plt.axis('square')


# ## Reflect

# In[13]:


A = np.array([[-1,0],[0,-0.5]])
print(A)

newdata = A.dot(data)
plt.plot(newdata[0,:], newdata[1,:], 'bo')
plt.plot(data[0,:], data[1,:], 'go')
plt.axis('square')


# ## Shrink

# In[ ]:


A = np.array([[0.5,0],[0,0.3]])
print(A)

newdata = A.dot(data)
plt.plot(newdata[0,:], newdata[1,:], 'bo')
plt.plot(data[0,:], data[1,:], 'go')
plt.axis('square')


# ## Expand

# In[17]:


A = np.array([[2,0],[0,1.5]])
print(A)

newdata = A.dot(data)
plt.plot(data[0,:], data[1,:], 'go')
plt.plot(newdata[0,:], newdata[1,:], 'bo')
plt.axis('square')


# ## Rotate

# In[19]:


theta = np.pi/2
A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
print(A)

newdata = A.dot(data)
plt.plot(newdata[0,:], newdata[1,:], 'bo')
plt.plot(data[0,:], data[1,:], 'go')
plt.axis('square')


# ## Shift
# 
# Be careful here!

# In[ ]:


A = np.array([[1,0],[0,1]])
print(A)

newdata = A.dot(data) + np.array([[0.6],[1.1]])
plt.plot(newdata[0,:], newdata[1,:], 'bo')
plt.plot(data[0,:], data[1,:], 'go')
plt.axis('square')

