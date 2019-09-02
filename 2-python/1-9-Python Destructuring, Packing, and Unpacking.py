#!/usr/bin/env python
# coding: utf-8

# We can "unpack" values with the following

# In[1]:


myfruit, yourfruit = ('apple', 'banana')
print(myfruit, yourfruit)


# And we can use the star mark (*) for variable length:

# In[2]:


myfruit, *otherfruits, yourfruit = ['apple', 'banana', 'orange', 'plum']
print(myfruit)
print(otherfruits)
print(type(otherfruits))
print(yourfruit)


# # functions

# Let's try unpacking with functions ... it's the same

# In[3]:


def give_me_fruits():
    return ['apple', 'banana', 'orange', 'plum']


# In[4]:


*myfruits, _ = give_me_fruits()
print(myfruits)


# Above we used `_` to denote a return that we want to just ignore (and not bind to a name).

# In[5]:


def some_fruits(fruit, *morefruits):
    print('The best fruit is %s' % fruit)
    for f in morefruits:
        print('...not %s' % f)


# In[6]:


some_fruits('apple', 'banana', 'orange', 'plum')


# In[7]:


def average_total(a,b):
    totalsum = a + b
    average = totalsum/2
    diff = a - b
    return average,totalsum, diff

#We can unpack values returned from a function
ave,tot,diff = average_total(3,5)
print(ave, tot, diff)

#We can unpack values using the star mark (*) for variable length
ave, *othervar, diff = average_total(3,5)
print(ave, diff)

ave, *_ = average_total(3,5)
print(ave)

