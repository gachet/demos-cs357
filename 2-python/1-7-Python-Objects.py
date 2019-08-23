
# coding: utf-8

# # Objects in Python

# Everything in Python is an 'object'.
# 
# Defining custom types of objects is easy:

# In[1]:


class Employee:
    empCount = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1
        
    def fire(self):
        self.salary = 0
        Employee.empCount -= 1
        


# * `empCount` is a class variable, and can be accessed inside and outside of the class as `Employee.empCount`
# 
# * Functions within the class (type) definition are called 'methods'.
# 
# * The first method `__init__` is a special method (called the class 'constructor' or initialization method.
# 
# * You define class methods like normal functions, except that the first argument to each method is the explicit `self` parameter.
# 
#     * Objects are created by 'calling' the type like a function.
#     * Arguments in this call are passed to the constructor

# In[2]:


# This creates the first employee Joe
joe = Employee("Joe",100000)
print('Employee name is ',joe.name)
print(joe.name, 'salary is $', joe.salary)


# In[3]:


# This will create the second employee Marc
marc = Employee("Marc",120000)
print('Employee name is ',marc.name)
print(marc.name, 'salary is $', marc.salary)


# In[4]:


print("Total employee number = ",Employee.empCount)


# We can add, remove and modify attributes at any time:

# In[5]:


joe.age = 28
joe.salary = 110000

print('Employee name is ',joe.name)
print(joe.name, 'salary is $', joe.salary)
print(joe.name, 'age is', joe.age)


# Let's fire Joe.

# In[6]:


joe.fire()


# In[7]:


joe.salary


# In[8]:


print("Total employee number = ",Employee.empCount)


# ## iclicker question

# What is the output of the following code snippet?

# 
# * a) Error message, because the function Change can't be called in the `__init__` function
# 
# * b) 'Old'
# 
# * c) 'New'
