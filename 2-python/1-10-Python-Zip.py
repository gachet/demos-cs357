
# coding: utf-8

# # Two helpful functions

# ## zip

# Suppose you have two lists:

# In[1]:


ids = ['a', 'b', 'c', 'd']
fruits = ['apples', 'bananas', 'oranges', 'grapes']
for c in zip(ids, fruits):
    print(c)


# ## enumerate

# suppose you had a single list:

# In[2]:


fruits = ['apples', 'bananas', 'oranges', 'grapes']
for i, f in enumerate(fruits):
    print("%d: the fruit is %s" % (i,f))

