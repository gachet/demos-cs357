#!/usr/bin/env python
# coding: utf-8

# # Python Introduction: Functions

# Functions help extract out common code blocks.
# 
# Let's define a function `print_greeting()`.

# In[1]:


def print_greeting():
    print("Hi there, how are you?")
    print("Long time no see.")


# And call it:

# In[2]:


print_greeting()


# That's a bit impersonal.

# In[3]:


def print_greeting(name):

    print("Hi there, {0}, how are you?".format(name))

    print("Long time no see.")


# In[4]:


print_greeting("Andreas")


# In[5]:


print_greeting()


# But we might not know their name. So we can set a default value for parameters.
# 
# (And we just changed the interface of `print_greeting`!)

# In[6]:


def print_greeting(name="my friend"):

    print("Hi there, {0}, how are you?".format(name))

    print("Long time no see.")


# In[8]:


print_greeting("Tim")


# Note that the order of the parameters does not matter

# In[11]:


def printinfo( name , age ):
    print("Name: ", name)
    print("Age: ", age)

printinfo(40,"Mariana")


# In[12]:


printinfo( age=8, name="Julia" )


# However the parameters "age" and "name" are both required above. What if we want to have optional parameters (without setting default values)?

# In[16]:


def printinfo( firstvar , *othervar ):
    print("First parameter:", firstvar)
    print("List of other parameters:")
    for var in othervar:
        print(var)

# Now you can call printinfo function
printinfo(10,20,30,50,60)


# A function can also return more than one parameter, and the results appear as a tuple:

# In[ ]:


def average_total(a,b):
    totalsum = a + b
    average = totalsum/2
    return average,totalsum

average_total(2,3)


# # Remember mutable and immutable types...

# Function parameters work like variables. So what does this do?

# In[ ]:


def my_func(my_list):
    my_list.append(5)
    print("List printed inside the function: ",my_list)
    
numberlist = [1,2,3]
print("List before function call: ",numberlist)
my_func(numberlist)
print("List after function call: ",numberlist)


# Can be very surprising! Here, we are maintaining reference of the passed object and appending values in the same object.
# 
# Define a better function `my_func_2`:

# In[ ]:


def my_func_2(my_list):
    
    my_list = my_list + [5]
    print("List printed inside the function: ",my_list)

    return my_list


numberlist = [1,2,3]
print("List before function call: ",numberlist)
new_list = my_func_2(numberlist)
#inside the function my_list = [1,2,3,5]
print("List after function call: ",numberlist)
print("Modified list after function call: ",new_list)


# Note that the parameter my_list is local to the function. 

# In[ ]:


def change_fruits(fruit):
    fruit='apple'
    print("I'm changing the fruit to %s" % fruit)


# In[ ]:


myfruit = 'banana'
change_fruits(myfruit)
print("The fruit is %s " % myfruit)


# **What happened?!** Remember that the input, `fruit` to `change_fruits` is bound to an object within the scope of the function:
#   * If the object is mutable, the object will change
#   * If the object is immutable (like a string!), then a new object is formed, only to live within the function scope.

# # Iclicker question:

# In[18]:


def do_stuff(a,b):
    a +=  [5]
    b += [8]


# In[28]:





# In[30]:


John = ['computer_science']
Tim = John
print(Tim,John)
add_minor(Tim)
print(Tim,John)
John = switch_majors(John)
print(Tim,John)

