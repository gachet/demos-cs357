#!/usr/bin/env python
# coding: utf-8

# # Python Introduction: A few more things

# Getting help:
# 
# 1) Use TAB in Jupyter or a quesiton mark

# In[1]:


a = [1,2,3]
get_ipython().run_line_magic('pinfo', 'a')


# 2) Using `pydoc3` on the command line.

# 3) Online at <http://docs.python.org/>

# ----

# **A few things to look up in a quiet moment**

# String formatting

# In[2]:


"My name is {0} and I like {1}".format("Andreas", "hiking")


# **or**

# In[3]:


"My name is %s and I like %s" % ("Andreas", "hiking")


# ---
# Dictionaries have *key*:*value* pairs of any Python object:

# In[4]:


prices = {"Tesla K40": 5000, "GTX Titan":1400}
prices["Tesla K40"]

