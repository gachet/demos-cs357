
# coding: utf-8

# # Objects and Naming
# 
# Understanding objects and naming in Python can be difficult.  In this demo we'll follow a post [Is Python call-by-value or call-by-reference? Neither.](https://jeffknupp.com/blog/2012/11/13/is-python-callbyvalue-or-callbyreference-neither/).

# First, let's make some objects.  *Everything in Python is an object*.

# In[ ]:


#So when we make a string it's an object. When we call it a variable name, it binds that name to the string object. 
fruit = 'apple'

#When we make a list, it will point to the object bound by fruit
lunch = []
lunch.append(fruit)

dinner = lunch
dinner.append('fish')

fruit = 'pear'

meals = [fruit, lunch, dinner]
print(meals)


# Let's check the object ids for both lists

# In[ ]:


print(id(lunch))
print(id(dinner))


# Notice what happens when we append to list that is bound to both `lunch` and `dinner`:

# In[ ]:


dinner.append('pasta')
print(lunch, dinner)


# In[ ]:


lunch.append('carrots')
print(lunch,dinner)


# # Mutable and Immutable
# 
# We've looked at mutable and immutable.  Tuples are an example of immutable objects.

# In[ ]:


fruits = ['apple', 'banana', 'orange']
veggies = ['carrot', 'broccoli']

food_tuple = (fruits, veggies)

print(food_tuple)

fruits.append('plum')

print(fruits)

print(food_tuple)


# So we can't change a tuple, but we can change the (mutable) things that a tuple element points to.

# # Iclicker question

# In[ ]:


Anna = ['electrical']
Julie = Anna
Julie += ['physics']
print(Anna)

