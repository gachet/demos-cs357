
# coding: utf-8

# In[1]:


import numpy.linalg as la
import numpy as np


# In[2]:


ndim = np.array([2,3,8,11,14])


# Let's perform linear solves for matrices with increasing size "n", for a problem in which we know what the solution would be.

# In[4]:


for nd in ndim:
    ## This is the vector 'x' that we want to obtain (the exact one)
    x = np.ones(nd)
    ## Create a matrix with random values between 0 and 1
    A = np.random.rand(nd,nd)
    ## We compute the matrix-vector multiplication 
    ## to find the right-hand side b
    b = A @ x
    ## We now use the linear algebra pack to compute Ax = b and solve for x
    x_solve = la.solve(A,b)
    ## What do we expect? 
    print("------ N =", nd, "----------")
    error = x_solve-x
    print("Norm of error = ", la.norm(error,2)) 


# In[6]:


print(x_solve)


# Now we will perform the same computation, but for a special matrix, known as the Hilbert matrix

# In[8]:


def Hilbert(n):
    
    H = np.zeros((n, n))    
    for i in range(n):        
        for j in range(n):        
            H[i,j] = 1.0/(j+i+1)    
    return H


# In[9]:


for nd in ndim:
    ## This is the vector 'x' that we want to obtain (the exact one)
    x = np.ones(nd)
    ## Create the Hilbert matrix
    A = Hilbert(nd)
    ## We compute the matrix-vector multiplication 
    ## to find the right-hand side b
    b = A @ x
    
    ## We now use the linear algebra pack to compute Ax = b and solve for x
    x_solve = la.solve(A,b)
    ## What do we expect? 
    print("------ N =", nd, "----------")
    error = x_solve-x
    print("Norm of error = ", la.norm(error,2)) 


# In[10]:


print(x_solve)


# ### What went wrong?

# ## Condition number

# The solution to this linear system is extremely sensitive to small changes in the matrix entries and the right-hand side entries. What is the condition number of the Hilbert matrix?

# In[11]:


for nd in ndim:
    ## This is the vector 'x' that we want to obtain (the exact one)
    x = np.ones(nd)
    ## Create the Hilbert matrix
    A = Hilbert(nd)
    ## We compute the matrix-vector multiplication 
    ## to find the right-hand side b
    b = A @ x
    ## We now use the linear algebra pack to compute Ax = b and solve for x
    x_solve = la.solve(A,b)
    ## What do we expect? 
    print("------ N =", nd, "----------")
    error = x_solve-x
    print("Norm of error = ", la.norm(error,2)) 
    print("Condition number = ", la.cond(A,2))


# ## Residual

# In[ ]:


for nd in ndim:
    ## This is the vector 'x' that we want to obtain (the exact one)
    x = np.ones(nd)
    ## Create the Hilbert matrix
    A = Hilbert(nd)
    ## We compute the matrix-vector multiplication 
    ## to find the right-hand side b
    b = A @ x
    ## We now use the linear algebra pack to compute Ax = b and solve for x
    x_solve = la.solve(A,b)
    ## What do we expect? 
    print("------ N =", nd, "----------")
    error = x_solve-x
    residual = A@x_solve - b
    print("Error norm = ", la.norm(error,2)) 
    print("Residual norm = ", la.norm(residual,2)) 
    print("Condition number = ", la.cond(A,2))


# In[ ]:


x_solve


# ## Rule of thumb

# In[ ]:


for nd in ndim:
    ## This is the vector 'x' that we want to obtain (the exact one)
    x = np.ones(nd)
    ## Create the Hilbert matrix
    A = Hilbert(nd)
    ## We compute the matrix-vector multiplication 
    ## to find the right-hand side b
    b = A @ x
    ## We now use the linear algebra pack to compute Ax = b and solve for x
    x_solve = la.solve(A,b)
    ## What do we expect? 
    print("------ N =", nd, "----------")
    error = x_solve-x
    residual = A@x_solve - b
    print("Error norm = ", la.norm(error,2)) 
    print("|dx| < ", la.norm(x)*la.cond(A,2)*10**(-16))
    print("Condition number = ", la.cond(A,2))

