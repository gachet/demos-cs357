
# coding: utf-8

# # Numpy Demo
# We'll go through some examples here.  There are also plenty of other guides online:
# * [Numpy quickstart tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)
# * [Numpy for MATLAB users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html) -- great reference if you already know MATLAB
# * [100 Numpy Exercises](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md)
# * [101 Numpy Exercises for Data Analysis](https://www.machinelearningplus.com/python/101-numpy-exercises-python/)
# * [Numpy Exercises, Practice, Solution](https://www.w3resource.com/python-exercises/numpy/index.php)
# 
# To start working with any package, need to import it.

# In[ ]:


import numpy as np


# # Motivation
# Arrays are faster and more efficient than lists when working with numerical data.

# ## Matrix Multiplication - Pure Python

# In[ ]:


import random
import time

start = time.time()

n = 100

A = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(random.random())
    A.append(row)
    
B = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(random.random())
    B.append(row)
    
C = []
for i in range(n):
    row = []
    for j in range(n):
        sum = 0
        for k in range(n):
            sum += A[i][k] * B[k][j]
        row.append(sum)
    C.append(row)
    
stop = time.time()
print(stop - start)


# ## Matrix Multiplication - Numpy

# In[ ]:


start = time.time()

n = 100

A = np.random.random((n, n))
B = np.random.random((n, n))
C = A @ B

stop = time.time()
print(stop - start)


# # Creating Arrays
# Numerous ways of creating arrays available.

# * Creating arrays from a list

# In[ ]:


vals_list = [1, 3, 2, 8]
vals_array = np.array(vals_list)

print("vals_list: ", vals_list)
print("vals_array: ", vals_array)


# In[ ]:


# don't need to create separate variable
vals_array = np.array([1,3,2,8])
print(vals_array)


# You can also change it back to a list:

# In[ ]:


print(vals_array.tolist())


# * Creating arrays using built-in functions
#     * [np.arange()](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.arange.html)
#     * [np.linspace()](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linspace.html)
#     * How do we know how to call them?
#         * See documentation
#         * Jupyter help

# ## Jupyter help - Two different ways

# In[ ]:


help(np.arange)


# In[ ]:


get_ipython().magic('pinfo np.arange')


# ## np.arange

# In[ ]:


# create array with integers 0,1,...,9
start = 0
end = 10 # needs to be 1 more than 9!
print(np.arange(start,end))


# In[ ]:


# can omit start if you want arange to start at 0
end = 12
print(np.arange(end))


# In[ ]:


# can take bigger step sizes than 1
# create array with integers 5, 9, 13, 17
start = 5
end = 21  # this is 17 + 4
step = 4
print(np.arange(start,end,step))


# In[ ]:


# don't need to go all the way to 21, 18 would be fine
start = 5
end = 18
step = 4
print(np.arange(start,end,step))


# ## np.linspace

# In[ ]:


start = 0
end = 1
num_points = 11
print(np.linspace(start,end,num_points))
print()
print(np.linspace(0,28,12))


# ## Initialize arrays with different values

# In[ ]:


print(np.zeros(5))
print()
print(np.ones(7))
print()
print(np.empty(3))


# ## Higher dimensional arrays

# Make a $5\times 5$ matrix of all zeros:

# In[ ]:


dim = (5,5) # must be a tuple!
print(np.zeros(dim))
print()
print(np.zeros((5,5)))


# Make a 3D array of shape $4\times 3 \times 2$ with all ones:

# In[ ]:


dim = (4,3,2)
print(np.zeros(dim))


# Also works with `np.empty`...

# # Data Types
# As we saw, Python is dynamically typed.  Types are changed automatically as needed.  And, lists can hold anything.  A single list could hold strings and integers.
# 
# What about arrays?
# Numpy arrays are statically typed.
# 
# So, what are the data types of the arrays we created above?  What are the available datatypes?  How do we specify what datatype we want? 

# In[ ]:


vals_list = [1,3,2,8]
vals_array = np.array(vals_list)
vals_arrayf = np.array(vals_list, dtype=np.float64)

print("vals_array: ", vals_array)
print("vals_arrayf: ", vals_arrayf)

print(type(vals_list))
print(type(vals_array))
print(type(vals_arrayf))


# The `dtype` argument is valid for most array-creation functions, including
# `numpy.zeros`, `np.ones`, and `np.arange`.

# In Python3, the `dtype` of an array that results from mathematical operations will
# automatically adjust to whatever is sensible.

# In[ ]:


print('integers: ', vals_array)
print('more integers: ', vals_array * 3)
print('floats: ', vals_array / 3)


# You can also copy an array and change the `dtype`.

# In[ ]:


arr = np.arange(10.0) # not an integer!
x = arr.astype(int)
print('arr: ', arr)
print('x: ', x)


# # Accessing Array Elements
# Now that we actually have arrays, how do we get things from them?
# Indexed from 0, bracket notation for accessing

# In[ ]:


vals_arrayf = np.array([1, 3, 2, 8, 24, 0, -1, 12])


# In[ ]:


print(vals_arrayf)
print()
print(vals_arrayf[0]) # this selects 0th element
print(vals_arrayf[3])


# Negative accessing is also allowed.

# In[ ]:


print(vals_arrayf)

print(vals_arrayf[-1])
print(vals_arrayf[-3])


# What if I want a section of an array?  **Array slicing**.

# In[ ]:


start_index = 1
end_index = 4  # will stop BEFORE this index - think about np.arange
print(vals_arrayf)
print()
print(vals_arrayf[start_index:end_index])
print(vals_arrayf[1:2])


# In[ ]:


print(vals_arrayf)
print()
print(vals_arrayf[1:1]) # this will get you all elements strictly between 1 and 1... there aren't any!


# In[ ]:


start = 2
end = 37 # going too far is fine
print(vals_arrayf)
print()
print(vals_arrayf[start:end])


# If you start at the beginning, no need to put in 0:

# In[ ]:


print(vals_arrayf[0:3])
print(vals_arrayf[:3])


# Similar if you want to end at the last element:

# In[ ]:


print(vals_arrayf[1:8])
print(vals_arrayf[1:])


# You can use negative indices too!

# In[ ]:


print(vals_arrayf)
print()
print(vals_arrayf[2:-1])
print()
print(vals_arrayf[:-2])


# In addition to a start and end, you can also choose a step for the slice.

# In[ ]:


start = 0
end = 6
step = 2
print(vals_arrayf)
print()
print(vals_arrayf[start:end:step])


# These next two calls do the same thing:

# In[ ]:


print(vals_arrayf[0:8:2])
print(vals_arrayf[::2])


# What are these next two examples doing?

# In[ ]:


print(vals_arrayf)
print()
print(vals_arrayf[1::2])
print(vals_arrayf[::-1])


# # Copies vs. Views (Accidentally changing your array)

# You need to be careful with `numpy` arrays if you are
# * trying to copy part of an array, or
# * passing an array to a function
# 
# You might be in for a nasty surprise if you change an element.

# In[ ]:


simple = np.arange(5)
small = simple[:2]
print(simple)
print('')
print(small)
print('')

small[0] = 7
print(small)
print('')
print(simple)  # shouldn't have changed, right?


# This happens because `small` is something called a "view" of
# `simple`, rather than a copy. This helps `numpy` save memory and
# speed up your program, but it can lead to tricky bugs if it
# is not your intent. In general, it can be difficult to tell
# whether something will be a view or a copy.
# 
# Functions also do not make copies of their input arrays.

# In[ ]:


def foo(x):  # notice that x is not returned
    x[0] = 100


foo(simple)
print(simple)


# If you think you are accidentally changing your array elsewhere in your code,
# you can copy it to be on the safe side. This slow your program down
# and use more memory, but it can help debugging and save a lot of headaches.

# In[ ]:


simple = np.arange(5)
print('before:')
print(simple)

my_copy = simple[:2].copy()
my_copy[1] = 10

foo(simple.copy())

print('after:')
print(simple)


# # Multi-dimensional Arrays

# *Note:* There is a `numpy.matrix` class, but you should **avoid** using it.
# Use two-dimensional arrays instead.

# How do we create multi-dimensional arrays?

# In[ ]:


# Creating from multi-dimensional lists
mat = np.array([[1,4,8],[3,2,9],[0,5,7]], float)
print(mat)
print('')


# Exercise: Define the following matrix
# \begin{bmatrix}
# 4 & 2.2 & 9 & 0 & 0.5\\
# 0 & 0   & -1 & 1 & 1\\
# 3 & -1 & 2 & 0 & 100
# \end{bmatrix}

# In[ ]:


# Creating special matrices
print(np.zeros((2,3), dtype=float)) # you already saw this...but this time we're specifying the type
print('')
print(np.zeros_like(mat))
print('')
# np.zeros_like creates a matrixsame shape, dimension, datatype as existing matrix
print(np.identity(3, dtype=float))


# How do we access multi-dimensional arrays?

# In[ ]:


print(mat)
print(mat[1,2])


# In[ ]:


print(mat[0,0], mat[1,1], mat[2,2])


# What is happening here?

# In[ ]:


print(mat)
print()
print(mat[2])


# Can do the same thing with array slicing:

# In[ ]:


print(mat[2,:])


# What's happening here?

# In[ ]:


print(mat[:,1])
print()
# force it to be a column vector...
print(mat[:,1:2])
print()
print(mat[:,[1]])


# What about this?

# In[ ]:


print(mat[:,:2])


# ## Exercise: Create a 2D array with 1 on the border and 0 inside
# For example,
# 
# 
# $\begin{bmatrix} 1 & 1 & 1 & 1 & 1\\
# 1 & 0 & 0 & 0 & 1\\
# 1 & 0 & 0 & 0 & 1\\
# 1 & 0 & 0 & 0 & 1\\
# 1 & 1 & 1 & 1 & 1
# \end{bmatrix}$

# In[ ]:


# you can do this with np.ones or np.zeros and slicing
N = 8  # make a matrix of size NxN


# What if we want an array of a different shape?
# This can be a convenient way of initializing matrices.

# In[ ]:


arr = np.arange(8)
two_four = arr.reshape(2, 4)
four_two = arr.reshape(4, 2)
eight_none = four_two.flatten()
print('array:')
print(arr.shape)
print('')
print('2 x 4:')
print(two_four)
print('')
print('4 x 2:')
print(four_two)
print('')
print('back to array:')
print(eight_none)
print(eight_none.shape)


# # Array functions
# We'll go through some array functions here.  There are plenty more available.  Best way to find the function you want is to search on Google for what you want and find the documentation for it (there is probably a function that does what you want to do).

# In[ ]:


new_mat = mat[:,:2]
print(new_mat)


# Shape of an array

# In[ ]:


print(new_mat.shape) # not actually a function, but an "attribute"


# What about sorting an array?
# Two different methods, 
# [np.sort()](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sort.html) or 
# [myarray.sort()](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.sort.html).
# One is a numpy function (called as np.sort()) and returns a copy of the array in sorted order.
# The other one is a function of the array and sorts the array in place.
# 
# **Important point:**
# * Some functions operate in place, others return copies.
# * How do you know which you are using?  Look at the documentation.

# In[ ]:


# Sort array
vals_arrayf = np.array([1, 3, 2, 8, 24, 0, -1, 12])

print(np.sort(vals_arrayf)) # returns a copy
print(vals_arrayf)
vals_arrayf.sort() # inplace
print(vals_arrayf)


# In[ ]:


# Checking if items are in an array
print(9 in mat)
print(9 in vals_arrayf)


# Lots of elementwise operations happen automatically with arrays. These include:
# * addition
# * subtraction
# * multiplication
# * division
# * comparisons

# In[ ]:


mat_2 = np.array([[1,3],[2,5]], float)
id_2 = np.identity(2, float)

print(mat_2)
print('')
print(id_2)
print('')
print('sum:')
print(mat_2 + id_2)
print('')
print('difference:')
print(mat_2 - id_2)
print('')
print('product:')
print(mat_2 * id_2)  # NOT matrix multiplication
print('')
print('quotient:')
print(id_2 / mat_2)
print('power:')
print(mat_2**3)


# Can compare values too!

# In[ ]:


print(mat_2 == id_2)
print(' ')
print(mat_2 > id_2)


# In[ ]:


# Other Functions
print(np.exp(id_2))
print()
print(np.abs(vals_arrayf))
print()
print(np.log2(mat_2))
print()
print(np.reciprocal(mat_2))


# In[ ]:


# Trig Functions
print(np.sin(mat_2))
print(np.tan(id_2))


# In[ ]:


# Rounding
print(np.round(np.sin(mat_2), 2))


# Can also perform operations between arrays and numbers:

# In[ ]:


print(mat_2)
print()
print(mat_2 - 3) # subtract 3 from every element
print()
print(mat_2*8) # multiply every element by 8


# # Exercise: 
# Given a numpy array $x$, compute an array that applies the following function to $x$:
# 
# $\begin{equation}
# e^{-|x|^3} + \sin(5x) + \cos(x + 3)
# \end{equation}$

# In[ ]:


x = np.linspace(-1,1,30)



# Recall comparing arrays:

# In[ ]:


print(mat_2)
print(id_2)
print()
print(mat_2 == id_2)
print(' ')
print(mat_2 > id_2)


# Can compare arrays and actually use the resulting boolean array to manipulate the entries of another array

# In[ ]:


z = mat_2 > id_2
print(z)
print()
print(mat_2)
print()
print(mat_2 * z)


# This might help you understand what happened:

# In[ ]:


print(z*1)


# # Exercise:  
# The Rectified Linear Unit (ReLU) is a function that is frequently used in machine learning, especially in the context of deep neural networks.  It is defined as:
# 
# $\begin{equation}
# \text{ReLU}(x) = \max\{0,x\}
# \end{equation}$
# 
# That is if $x < 0$ it returns zero, and if $x\geq 0$ it returns $x$.
# 
# Given a 1D array $x$, compute its transformation under the ReLU function using comparison.  *Hint*: just like how you can add a single number to every element of an array, you can also compare a single number to every element.

# In[ ]:


x = np.linspace(-1,1,30)



# # Broadcasting (Element-wise operations on arrays of different shapes)

# Not necessary for this course, but check it out if you're interested.  Every broadcast operation can be done using loops, but broadcasting is faster.  You will get by just fine in CS 357 using loops

# See [A Gentle Introuction to Broadcasting with Numpy Arrays](https://machinelearningmastery.com/broadcasting-with-numpy-arrays/) for a detailed explanation

# The simplest case of broadcasting is adding a single number to every element of an array.  Here is the mathematically correct way of adding a number to every element of an array

# In[ ]:


bmat = np.arange(12).reshape(4, 3)
print(bmat)
print()
z = 3*np.ones_like(bmat)  # what is this doing?
print(z)
print()
print(bmat + z)


# But we saw can just add a number directly...Numpy is **broadcasting** the value

# In[ ]:


print(bmat)
print()
z = 3
print(z)
print()
print(bmat + z)


# **Advice**: Get the hang of Python and Numpy, and worry about broadcasting later.  Just know that you can add a single number to an array, and there is something going on behind the scenes

# # Reduction Operations

# There are other operations that do not return an array of the same shape as the input.
# For example, you can find out the minimum or maximum value in the entire array,
# or the sum of all entries.

# In[ ]:


bmat = np.array([6, 7, -12, 0, 3, 4, 21, 1, 1, 0, 2, 5]).reshape(4,3)


# In[ ]:


print(bmat)
print()
print(bmat.min())
print(bmat.max())
print(bmat.sum())


# What if I want the smallest number in every row?
# All of these reduction operations take an optional `axis` argument that allows
# us to target a particular dimension of the array.

# In[ ]:


print('row minimum:')
print(bmat.min(axis=1))
print('column maximum:')
print(bmat.max(axis=0))
print('row sum:')
print(bmat.sum(axis = 1))


# Notice that when we pass an `axis` argument, we lose that dimension of our
# array, but the shape is otherwise unchanged. So, a (4, 3) array becomes
# a (3,) array if we pass `axis=0`, and it becomes a (4,) array if we
# pass `axis=1`.

# Some more reductions...

# In[ ]:


print(bmat)
print()
print('mean:')
print(bmat.mean())
print()
print('column mean:')
print(bmat.mean(axis = 0))


# In[ ]:


print(bmat)
print()
print('product:')
print(bmat.prod())
print('column product:')
print(bmat.prod(axis = 0))


# # Treating Arrays as Matrices and Vectors

# If `*` is elementwise multiplication, how do we do matrix multiplication?

# In[ ]:


# Matrix Multiplication and Dot Product
print(np.dot(mat_2, id_2))
print('')
print(mat_2 @ id_2)
print('')
print(np.dot(vals_arrayf, np.array([0,2,6,1, 1, 2, 3, 4])))


# In[ ]:


# Matrix transpose
print(mat_2)
print()
print(np.transpose(mat_2))
print()
print(mat_2.T)


# # Numpy constants
# A [list](https://docs.scipy.org/doc/numpy/reference/constants.html) of Numpy constants

# In[ ]:


print(np.pi) # the famous irrational number
print(np.e)  # euler's number = exp(1)
print(np.inf) # infinity
print(np.NINF) # negative infinity
print(np.nan)  # 'not a number'


# # Random Numbers

# You will often be asked to generate random numbers.
# `numpy` can generate numbers from a variety of distributions,
# and it can generate lots of them at once and put them in a convenient shape.
# 
# The [np.random](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html) documentation gives a helpful overview.
# 
# Some of the more common functions you might use are
# [np.random.rand](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.rand.html#numpy.random.rand) (uniform),
# [np.random.randn](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.randn.html#numpy.random.randn) (normal),
# and [np.random.randint](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.randint.html#numpy.random.randint) (integers).
# All of these routines give you the option of generating an array of a specified shape.

# In[ ]:


uniform_nums = np.random.rand(10)
print(uniform_nums)
print('')
normal_nums = np.random.randn(3, 5)
print(normal_nums)
print('')
integers = np.random.randint(0, 10, (4, 2))
print(integers)
print('')


# # What Else is There in Numpy
# There's so many more functions in Numpy!  Read documentation and **Google** things.  Someone probably asked your question on Stack Exchange or Stack Overflow already!

# This tutorial didn't really get in to any of the functions in 
# [numpy.linalg](http://docs.scipy.org/doc/numpy/reference/routines.linalg.html).
# We'll see a lot of those functions in class.

# # More Exercises:
# 1. We'll do this one together...
# 
# Let $B$ be a $4x4$ matrix and apply the following operations to it (in this order):
#     * Double the first column
#     * Halve the third row
#     * Add the third row to the first row
#     * Interchange the first and last columns
#     * Subtract the second row from each of the other rows
#     * Replace column 4 by column 3
#     * Delete the 1st column (so the matrix is now 4 by 3) - see np.delete

# In[ ]:


B = np.random.randint(-4,4,(4,4)).astype(float)
print(B)


# In[ ]:


def my_function(B):
    ...


# 2. Generate a 1D Numpy array of 20 integers in the range $[2, 12)$.  Count how many are greater than 8 without using a loop.  Also return an array of the same size that has the same values as the original array when they are greater than 8 and zero otherwise.
# 
# 
# 3. Repeat the exercise with ''greater than'' replaced with ''greater than or equal to''

# 4. The Frobenius inner product of two matrices can be defined as $\text{tr}(\mathbf{A}^T\mathbf{B})$ where ''tr'' refers to the [trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra) (just the sum of the diagonals).  Write a function to compute the Frobenius inner product of two matrices.
# 
#     * There's a numpy function called ``np.trace`` that makes this easy
#     * Try to do it without ``np.trace``.  You can use ``np.sum``, and a function called ``np.diag``

# In[ ]:


def Frobenius(A,B):
    ...


# 5. Generate a 10 by 10 matrix of normally distributed values.  Write a function that returns the column index of the column with the largest mean
# 
#     * *Hint*: check out ``np.argmax``

# In[ ]:


A = ...

def column_largest_mean(A):
    ...


# Work through these!
# * [100 Numpy Exercises](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md)
# * [101 Numpy Exercises for Data Analysis](https://www.machinelearningplus.com/python/101-numpy-exercises-python/)
# * [Numpy Exercises, Practice, Solution](https://www.w3resource.com/python-exercises/numpy/index.php)
# 
# Come to office hours or ask on Piazza if you want clarification about what's happening in these questions
