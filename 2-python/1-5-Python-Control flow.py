#!/usr/bin/env python
# coding: utf-8

# # Python Introduction: Control Flow

# `for` loops in Python always iterate over something list-like:

# In[2]:


for i in range(3,10):

    print(i)
    
    


# **Note** that Python does block-structuring by leading spaces.
# 
# Also note the trailing "`:`".

# ---
# `if`/`else` are as you would expect them to be:

# In[ ]:


for i in range(10):
    if i % 3 == 0:
        print("{0} is divisible by 3".format(i))
    else:
        print("{0} is not divisible by 3".format(i))


# In[ ]:


print("My name is %s" % 'Luke')
print("My name is {}".format('Luke'))


# `while` loops exist too:

# In[ ]:


i = 0
while True:
    i += 1
    if i**3 + i**2 + i + 1 == 3616:
        break

print("SOLUTION:", i)


# ----
# Building lists by hand can be a little long. For example, build a list of the squares of integers below 50 divisible by 7:

# In[ ]:


mylist = []

for i in range(50):

    if i % 7 == 0:

        mylist.append(i**2)


# In[ ]:


mylist


# Python has a something called *list comprehension*:

# In[3]:


mylist = [i**2 for i in range(50) if i % 7 == 0]
print(mylist)


# Dictionaries

# In[5]:


#mydict = {key: value}
mydict = {'Luke': 15,'Mariana' : 22}
print(mydict["Mariana"])


# In[ ]:


string = "Batman"
mydict = {key:ord(key) for key in string}


# In[ ]:


print(mydict)

