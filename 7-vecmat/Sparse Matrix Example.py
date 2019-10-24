
# coding: utf-8

# In[1]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sparse

# print(plt.style.available) # uncomment to print all styles
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (10,10)
get_ipython().magic('matplotlib inline')


# <img src="sparse.png" alt="Sparse" style="width: 300px;"/>

# ## Create a Sparse Matrix in COO

# In[2]:


data = [1.9, -5.2, 0.3, 9.1, 4.4, 5.8, 3.6, 7.2, 2.7]
i    = [  0,    0,   1,   1,   2,   2,   2,   3,   3]
j    = [  1,    3,   0,   2,   0,   1,   2,   2,   3]
A = sparse.coo_matrix((data, (i, j)))

print(A)
print(A.todense())


# In[ ]:


data = [-5.2, 1.9, 0.3, 9.1, 4.4, 5.8, 3.6, 7.2, 2.7]
i    = [   0,   0,   1,   1,   2,   2,   2,   3,   3]
j    = [   3,   1,   0,   2,   0,   1,   2,   2,   3]
A = sparse.coo_matrix((data, (i, j)))
print(A.todense())


# In[ ]:


print(A.data)
print(A.data.dtype, 'Length: ', len(A.data))
print('-')
print(A.row)
print(A.row.dtype, 'Length: ', len(A.row))
print('-')
print(A.col)
print(A.col.dtype, 'Length: ', len(A.row))


# ## Convert to CSR

# In[ ]:


A = A.tocsr()
print(A)
print(A.todense())


# In[ ]:


print(A.data)
print(A.data.dtype, 'Length: ', len(A.data))
print('-')
print(A.indptr)
print(A.indptr.dtype, 'Length: ', len(A.indptr))
print('-')
print(A.indices)
print(A.indices.dtype, 'Length: ', len(A.indices))


# ## Try some timings: small, `Harvard500`

# In[ ]:


import scipy.io as sio
d = sio.loadmat('./Harvard500.mat')
A = d['Problem'][0][0][2].tocsr()


# In[ ]:


A


# In[ ]:


plt.figure(figsize=(10,10))
plt.spy(A, ms=5)


# In[ ]:


A.shape[0]


# In[ ]:


v = np.random.rand(A.shape[0])
w = np.random.rand(A.shape[0])


# In[ ]:


get_ipython().magic('timeit v = A * w')


# In[ ]:


Adense = A.todense()


# In[ ]:


get_ipython().magic('timeit v = Adense.dot(w)')


# ## Medium `wb-cs-stanford`

# In[ ]:


d = sio.loadmat('./wb-cs-stanford.mat')
A = d['Problem'][0][0][2].tocsr()


# In[ ]:


plt.figure(figsize=(10,10))
plt.spy(A, ms=5)


# In[ ]:


A


# In[ ]:


v = np.random.rand(A.shape[0])
w = np.random.rand(A.shape[0])


# In[ ]:


get_ipython().magic('timeit v = A * w')


# In[ ]:


Adense = A.todense()


# In[ ]:


get_ipython().magic('timeit v = Adense.dot(w)')


# ## Large `email-Enron`

# In[ ]:


d = sio.loadmat('./email-Enron.mat')
A = d['Problem'][0][0][2].tocsr()


# In[ ]:


plt.figure(figsize=(10,10))
plt.spy(A, ms=5)


# In[ ]:


A


# In[ ]:


v = np.random.rand(A.shape[0])
w = np.random.rand(A.shape[0])


# In[ ]:


get_ipython().magic('timeit v = A * w')


# In[ ]:


Adense = A.todense()


# In[ ]:


get_ipython().magic('timeit v = Adense.dot(w)')


# ## Does 36692 sound "large"?
