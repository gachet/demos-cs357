
# coding: utf-8

# In[ ]:


# Datatypes

# Integers, Floats, and Complex Numbers

# Basic operators: "+", "-", "*", "%", "**", "/", "//"
# Some operators only work with integers

print(5 + 4) 
print(7 - 8)
print(10 * 6)
print(10 % 7)


# In[ ]:


# Try it

# Calculate 10 + 3.333

# Calculate 4 raised to the fifth power

# Calculate 10 float-divided by 6

# Calculate 10 integer-divided by 6


# In[ ]:


# Booleans

# Boolean literals: "True", "False"

# Boolean operators: "and", "(inclusive) or", "not"
print(True)
print(True and False)


# In[ ]:


# Try it

# Print the result of ~T

# Print the result of T ∨ F

# Print the result of "~T ∧ F"


# In[ ]:


# Strings

print('abc')
print("abc")
print('a "quoted" string' + "another \"quoted string\" " + "can't")
print('abc' + 'a')

print("""
         This
         is
         a 
         multiline
         string.""")
print("""
         So
         is
         this.
         """)

# Multiline strings are functionally multiline comments to
print("Code I want to run")

'''
print("Code I want not to run")
print("More code I want not to run")
'''


# In[ ]:


# Try it:

# Write two multi-line strings, concatenate them, and print the result


# In[ ]:


# Variables (weakly typed)

x = 17
print(x)

x = x + 12
print(x)

y = x / 3.2
print(y)

z = 'grape'
z = z + str(y)
print(z)


# In[ ]:


# Try it:

# Repeat the the string concatenation exercise above using variables to store each string


# In[ ]:


# Typecasting

print(str(5))
print(float(7))


# In[ ]:


# Try it:

# Cast 7.3 to an int and print the result

# Cast the string "3.2" to a float and print the result


# In[ ]:


# Print statements

five = 5.333
name = "bob"
print(name)
print(name + str(five))
print(name, five)

#Remove string formatting
print("{}".format(five))
print("{}".format(name))
print("{} is {:.0f} years old.\n".format(name, five))
print("Name:\tAge:\n{}\t{:.1f}\n".format(name, five))

# C-style print strings work as well
print("Name:\tAge:\n%s\t%.1f\n" % (name, five))


# In[ ]:


# Lists

# Creating a list.

a = [1, 2, 3, 4]
b = ['hello', 1, 2.5]
c = [[1, 2, 3],
     [4, 5, 6]]


# In[ ]:


# Try it:

# Create list 'f' containing the characters of your first name

# Create a list 'l' containing the characters of your last name

# Create a list 'name' that contains 'f' and 'l'


# In[ ]:


# Accessing a list.

print(a[0])
print(a[-1])  # last element
print(c[0])
print(c[0][2])


# In[ ]:


# Try it:

# Index into 'name' and extract the last character of your first name


# In[ ]:


# What happens if we access an element out of the bounds of our array?

print(a[4])  # out of range


# In[ ]:


# List slicing.

print(a[1:3])
print(a[1:])
print(a[:2])


# In[ ]:


# Try it:

# Slice into 'name' and print the first two characters of your first name and last name


# In[ ]:


# Editing a list.

a.append(6)  # adds to the end
print(a)
a[4] = 5  # sets item at position 4
print(a)
a.remove(3)  # removes the first 3 in the list
print(a)
del a[2]  # removes the item at position 2
print(a)


# In[ ]:


# Try it:

# Append the first character of your last name to 'f' and print 'f'

# Remove the character you added and print 'f' again


# In[ ]:


# Other list functions.

print(a)
print(len(a))
print(a.index(1))  # the index of the first 3
print(a.count(5))
a.reverse()  # these are in-place operations
print(a)
a.sort()
print(a)


# In[ ]:


# Try it:

# Reverse 'f'

# Sort 'l'

# Count the occurrences of "a" in your last name


# In[ ]:


# Data is not necessarily copied

a = [0, 1, 2, 3, 4]
b = a
b[2] = 6

print(a)
print(b)

c = a.copy()
c[4] = 7

print(a)
print(c)


# In[ ]:


# Try it:

# Make a copy of 'l' called 'l_copy' and change the first element of "l_copy" to "z"

# Print 'l' and 'l_copy'. Did 'l' change?


# In[ ]:


# Tuples

a = (1, 2, 3)
b = (['hi', 3], 5, 7)
print(a[2])
x, y = ('diameter', 4 + 5)
print(x)
print(y)


# In[ ]:


# Try it:

# Create a tuple 'f_tup' with the characters of your first name.

# Try to change the first entry of 'f_tub'. What happens?


# In[ ]:


a[2] = 5  # error: cannot change a tuple


# In[37]:


# Dictionaries

# Map an ID to a value

# Create an empty dictionary
d = {}
print(d)
d = dict()
print(d)

# Create a full dictionary
dnd_character = {"name": "Monty", "hitpoints": 23, "strength": 12, "alignment": "lawful good"}
print(dnd_character)

# Add an entry
dnd_character['wisdom'] = 16
print(dnd_character)

# Check if a key is in a dictionary
print('wisdom' in dnd_character)

# Lookup a value
print(dnd_character['wisdom'])

# Append/update dictionaries
more_stats = {"intelligence": 10, "charisma": 14}
dnd_character.update(more_stats)
print(dnd_character)

# Delete an entry
del dnd_character['strength']
print(dnd_character)


# In[ ]:


# Try it:

# Create a small dictionary that maps colors to an item you associate with that color
# Experiment with the above operations on your dictionary.


# In[ ]:


# Conditional statements

if True:
    print('True')
    print('Also True')
    
if False:
    print('False')
    
if True and False:
    print('False')
    print('Also False')
    
    
    
    print('Still False down here')
print('Not in conditional')


# In[ ]:


# Try it:

# Test if a key exists in your dictionary. If it does not exist then add the key and its value to the dictionary.


# In[ ]:


# Be careful with whitespace

if True:
  print('True')
    print('Tabbing is wrong')


# In[ ]:


# Multi-way decisions

if 5 > 4:
    print('This will be printed')
else:
    print('This will not be printed')
    
if 5 > 8:
    print('Not printed')
elif 9 > 8:
    print('Print this')
else:
    print('Not this')


# In[ ]:


# Try it

# Test if a key exists in your dictionary. 
# If it does not exist then add the key and its value to the dictionary.
# Otherwise print its value.


# In[ ]:


# Nested conditions

if 'abc' == 'def':
    if 5 > 4 and 5 >= 5:
        print('Not here')
    else:
        print('Definitely not here')
else:
    if not True:
        print('False')
    elif 6 != 6:
        print('False')
    else:
        print('Print this')    


# In[ ]:


# For loops

for name in ['Alice', 'Bob', 'Claire']:
    print('Hello from {}'.format(name))
    

print('\n\n')

loop_num = 0
ice_creams = ('Chocolate', 'Vanilla', 'Strawberry')
for flavor in ice_creams:
    print('I want some {} ice cream'.format(flavor))
    
    loop_num += 1
    print('We have looped {} times'.format(loop_num))


# In[ ]:


# Try it

# Iterate over 'f' and print each character


# In[ ]:


# Range Function

print(range(3))
print(range(2, 5))
print(range(0, 20, 2))


# In[ ]:


a = range(5)
print(a[0])
print(a[-1])
a.append(5)  # error: range returns a generator, not a list


# In[ ]:


# Range and lists

for i in range(3):
    print(i)

print('\n\n')

# Get the product of numbers one through ten
product = 1
for i in range(1, 11):
    product *= i
print(product)


# In[ ]:


# Try it

# Repeat the above exercise, but instead 
# iterate over the range of the length of 'f'.
# Index into 'f' to print each character.


# In[ ]:


# Nesting

for i in range(3):
    for j in range(2):
        print((i,j))


# In[ ]:


# While loops

val = 0
while val < 5:
    print('looping')
    val += 1
print(val)


# In[ ]:


# Iterate until a convergence tolerance is met

n = 0
sol = 2.
tol = 1e-5
curr = 0
while abs(sol - curr) > tol:
    curr += 0.5**n
    print(curr)
    n += 1
    
print('Converged in {} iterations'.format(n))


# In[ ]:


# Try it:

# Print the characters in 'f' again, but use a while-loop instead of a for-loop


# In[ ]:


# Functions

def greet(name):
    print('Hello {}! Welcome to CS 357!'.format(name))
    
greet('Alice')


# In[ ]:


def convert(deg, celsius=True):
    if celsius:
        temp = (9 / 5.) * deg + 32
    else:
        temp = (5 / 9.) * (deg - 32)
    
print('It is {} degrees Fahrenheit outside.'.format(convert(0)))
print('I happen to know that is {} degrees Celsius.'.format(convert(32, False)))


# In[ ]:


# What went wrong?  We forgot to return the value we wanted from the function.

def convert(deg, celsius=True):
    if celsius:
        temp = (9 / 5.) * deg + 32
    else:
        temp = (5 / 9.) * (deg - 32)
    
    return temp

    
print('It is {} degrees Fahrenheit outside.'.format(convert(0)))
print('I happen to know that is {} degrees Celsius.'.format(convert(32, False)))


# In[ ]:


#Try it:

# Write a function which takes a 24-hour time as an integer
# and returns the equivalent 12-hour time as a string.
# The first two digits are the hour and the last two digits
# are the minutes


# In[ ]:


# Math operations 

import numpy as np
print(np.sin(2*np.pi))
print(np.log(np.e))


# In[ ]:


# Try it:

# Calculate cos(sqrt(pi*e))


# In[ ]:


# Need to figure out how to use a function?

help(np.log)


# In[ ]:


get_ipython().magic('pinfo np.log')

