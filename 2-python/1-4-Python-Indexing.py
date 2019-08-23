
# coding: utf-8

# # Python Introduction: Indexing

# The `range` function lets us build a list of numbers.

# In[1]:


list(range(1,10))


# In[5]:


list(range(10, 20,1))


# Notice anything funny?
# 
# Python uses this convention everywhere.

# In[6]:


a = list(range(10, 20))
type(a)


# Let's talk about indexing.
# 
# Indexing in Python starts at 0.

# In[7]:


a[0]


# And goes from there.

# In[8]:


a[1]


# In[9]:


a[2]


# What do negative numbers do?

# In[10]:


a[-1]


# In[11]:


a[-2]


# You can get a sub-list by *slicing*.

# In[12]:


print(a)
print(a[3:7])


# Start and end are optional.

# In[13]:


a[3:]


# In[14]:


a[:3]


# Again, notice how the end entry is not included:

# In[ ]:


print(a[:3])
print(a[3])


# Slicing works on any sequence type! (`list`, `tuple`, `str`, `numpy` array)

# In[ ]:


a = "CS357"
a[-3:]

