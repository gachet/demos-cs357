#!/usr/bin/env python
# coding: utf-8

# # Python Introduction: Names and Values

# Define and reference a variable:

# In[ ]:





# In[ ]:





# No type declaration needed!
# 
# (But values still have types--let's check.)

# In[ ]:





# Everything in python is an object. Python variables are like *pointers*.
# 
# (if that word makes sense)

# In[ ]:


a = [1,2,3]


# In[ ]:


b = a


# In[ ]:





# In[ ]:





# In[ ]:





# You can see this pointer with `id()`.

# In[ ]:





# The `is` operator tests for object sameness.

# In[ ]:





# This is a **stronger** condition than being equal!

# In[ ]:


a = [1,2,3]
b = [1,2,3]
print("IS   ", a is b)
print("EQUAL", a == b)


# What do you think the following prints?

# In[ ]:


a = [1,2,3]
b = a
a = a + [4]
print(b)
print(a)


# In[ ]:


a is b


# Why is that?

# -----
# How could this lead to bugs?

# ----------
# * To help manage this risk, Python provides **immutable** types.
# 
# * Immutable types cannot be changed in-place, only by creating a new object.
# 
# * A `tuple` is an immutable `list`.

# In[ ]:


a = [1,2,3]
type(a)


# In[ ]:


a[2] = 0
print(a)


# In[ ]:





# Let's try to change that tuple.

# In[ ]:


# clear
a[2] = 0


# *Bonus question:* How do you spell a single-element tuple?

# In[ ]:





# String is also immutable type. 

# In[2]:


myName = 'Mariena'


# Note that myName is spelled incorrectly. We would like to change the letter "e" with a letter "a"

# In[3]:


myName[4]


# In[4]:


myName[4]='a'


# # Memory management

# In[4]:


a = "apple"
b = "apple"
print(id(a),id(b))
print (a is b)
print (a == b)


# <img src="PointToString.png",width=200>

# Note that "a" and "b" are bounded to the same object "apple", and therefore they have the same id. For optimization reasons, Python does not store duplicates of "simple" strings. But if the strings are a little more complicated...

# In[16]:


a = "Hello, how are you?"
b = "Hello, how are you?"
print(id(a),id(b))
print (a is b)
print (a == b)


# This is called interning, and Python does interning (to some extent) of shorter string literals (such as "apple") which are created at compile time. But in general, Python string literals creates a new string object each time (as in "Hello, how are you?"). Interning is runtime dependant and is always a trade-off between memory use and the cost of checking if you are creating the same string. 

# In general, integers, floats, lists, tuples, etc, will be stored at different locations, and therefore have different ids.

# In[5]:


a = 5000
b = 5000
print(id(a),id(b))
print (a is b)
print (a == b)


# In[6]:


a = (1,[2,3],4)
b = (1,[2,3],4)
print(id(a),id(b))
print (a is b)
print (a == b)


# Also for optimization reasons, Python will <strong>not</strong> duplicate integers between -5 and 256. Instead, it will  keep an array of integer objects for all integers between -5 and 256, so when you create an int in that range you actually just get back a reference to the existing object. 

# In[21]:


a = 256
b = 256
print(id(a),id(b))
print (a is b)
print (a == b)


# In[20]:


a = 257
b = 257
print(id(a),id(b))
print (a is b)
print (a == b)


# http://foobarnbaz.com/2012/07/08/understanding-python-variables/
