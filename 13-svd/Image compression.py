
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as pt
get_ipython().magic('matplotlib inline')


# In[2]:


from PIL import Image

with Image.open("quad.jpg") as img:
    rgb_img = np.array(img)
rgb_img.shape


# In[3]:


img = np.sum(rgb_img, axis=-1)
#img = rgb_img
img.shape


# In[4]:


pt.figure(figsize=(20,10))
pt.imshow(img, cmap="gray")


# In[6]:


u, sigma, vt = np.linalg.svd(img)
print(u.shape)
print(sigma.shape)
print(vt.shape)


# In[7]:


sigma[:20]


# In[8]:


pt.plot(sigma, lw=4)
pt.xlabel('singular value index')
pt.ylabel('singular values')


# In[9]:


pt.loglog(sigma, lw=4)
pt.xlabel('singular value index')
pt.ylabel('singular values')


# In[12]:


k=50
compressed_img = u[:,:k] @ np.diag(sigma[:k]) @ vt[:k,:]
pt.figure(figsize=(20,10))
pt.imshow(compressed_img, cmap="gray")


# In[13]:


original_size = img.size
compressed_size = u[:,:k].size + sigma[:k].size + vt[:k,:].size
print("original size: %d" % original_size)
print("compressed size: %d" % compressed_size)
print("ratio: %f" % (compressed_size / original_size))

