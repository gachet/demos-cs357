
# coding: utf-8

# In[ ]:


import numpy as np
import numpy.linalg as la

import scipy.optimize as sopt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
get_ipython().magic('matplotlib inline')


# # Optimization algorithms for ND problems

# We provide three example of functions. You will be able to observe difference convergence carachteristics among them.

# #### Function 1:
# $$ f(x,y) = 0.5 x^2 + 2.5 y^2 $$

# In[ ]:


def f1(x):
    return 0.5*x[0]**2 + 2.5*x[1]**2

def df1(x):
    return np.array([x[0], 5*x[1]])

def ddf1(x):
    return np.array([
                     [1,0],
                     [0,5]
                     ])


# #### Function 2:
# $$ f(x,y) = (x-1)^2 + (y-1)^2 $$

# In[ ]:


def f2(x):
    return (x[0]-1)**2 + (x[1]-1)**2

def df2(x):
    return np.array([2*(x[0]-1),2*(x[1]-1) ])

def ddf2(x):
    return np.array([
                     [2,0],
                     [0,2]
                     ])


# #### Function 3:
# $$ f(x,y) = 100 (y-x^2)^2 + (1-x)^2 $$

# In[ ]:


def f3(X):
    x = X[0]
    y = X[1]
    val = 100.0 * (y - x**2)**2 + (1.0 - x)**2
    return val

def df3(X):
    x = X[0]
    y = X[1]
    val1 = -400.0 * (y - x**2) * x - 2 * (1 - x)
    val2 = 200.0 * (y - x**2)
    return np.array([val1, val2])

def ddf3(X):
    x = X[0]
    y = X[1]
    val11 = -400.0 * (y - x**2) + 800.0 * x**2 + 2
    val12 = -400.0 * x
    val21 = -400.0 * x
    val22 = 200.0
    return np.array([[val11, val12], [val21, val22]])


# ### Helper functions for plotting

# In[ ]:


def plotFunction(f, interval=(-2,2), levels=20, steps=None, fhist=None):
    
    a,b = interval
    
    xmesh, ymesh = np.mgrid[a:b:100j,a:b:100j]
    fmesh = f(np.array([xmesh, ymesh]))
    
    
    fig = plt.figure(figsize=(16,4))

    ax = fig.add_subplot(131,projection="3d")
    ax.plot_surface(xmesh, ymesh, fmesh,cmap=plt.cm.coolwarm);
    plt.title('3d plot of f(x,y)')

    ax = fig.add_subplot(132)
    ax.set_aspect('equal')
    c = ax.contour(xmesh, ymesh, fmesh, levels=levels)

    plt.title('2d countours of f(x,y)')
    ax.clabel(c, inline=1, fontsize=10)
    
    if steps is not None:  
        plt.plot(steps.T[0], steps.T[1], "o-", lw=3, ms=10)
     
    if fhist is not None:
        ax = fig.add_subplot(133)
        plt.semilogy(fhist, '-o')
        plt.xlabel('iteration')
        plt.ylabel('f')
        plt.grid()


# In[ ]:


plotFunction(f1)


# In[ ]:


plotFunction(f2)


# In[ ]:


plotFunction(f3,levels=np.logspace(0,4,10))


# In[ ]:


def plotConvergence( steps, exact , r ):
       
    error = la.norm(np.array(steps) - np.array(exact),axis=1)
    ratio = []
    for k in range(len(error)-1):
        ratio.append( error[k+1]/error[k]**r )

    fig = plt.figure(figsize=(4,4))

    plt.plot(ratio, "o-", lw=3, ms=10)
    plt.ylim(0,2)


# # Steepest Descent

# In[ ]:


def steepestDescent(f,df,x0,maxiter,tol):

    # Line search function
    def f_line(alpha):
        fnew = f(x + alpha*s)
        return fnew
    
    steps = [x0]   
    x = x0
    fhist = [f(x)]
    
    # Steepest descent with line search
    for i in range(maxiter):

        # Get the gradient
        s = -df(x)

        # Line  search
        alpha_opt = sopt.golden(f_line)

        # Steepest descent update
        xnew = x + alpha_opt * s

        # Save optimized solution for plotting
        steps.append(xnew)
        
        fhist.append(f(xnew))

        # Check convergence
        
        if ( np.abs(fhist[-1] - fhist[-2]) < tol ):
            break

        x = xnew
        
    print('optimal solution is:', x)
        
    return steps, fhist, i   


# In[ ]:


# Initial guess
x0 = np.array([2, 2./5])
# Steepest descent
steps, fhist, iterations = steepestDescent(f1,df1,x0,50,1e-6)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f1,steps=np.array(steps),fhist=np.array(fhist))

plotConvergence( steps, [0,0] , 1 )


# In[ ]:


# Initial guess
x0 = np.array([-1.5, -1])
# Steepest descent
steps, fhist, iterations = steepestDescent(f2,df2,x0,50,1e-4)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f2,steps=np.array(steps),fhist=np.array(fhist))



# In[ ]:


# Initial guess
x0 = np.array([0, 1.75])
# Steepest descent
steps, fhist, iterations = steepestDescent(f3,df3,x0,1000,1e-6)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f3,steps=np.array(steps),levels=np.logspace(0,4,8), fhist=np.array(fhist))

plotConvergence( steps, [1 , 1] , 1 )


# In[ ]:


# Initial guess
x0 = np.array([-0.5, -1])
# Steepest descent
steps, fhist, iterations = steepestDescent(f3,df3,x0,1000,1e-6)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f3,steps=np.array(steps),levels=np.logspace(0,4,8), fhist=np.array(fhist))

plotConvergence( steps, [1 , 1] , 1 )


# # Newton's method

# In[ ]:


def NewtonMethod(f,df,ddf,x0,maxiter,tol):
    
    steps = [x0]   
    x = x0
    fhist = [f(x)]
    
    # Steepest descent with line search
    for i in range(maxiter):

        # Get the newton step
        s = la.solve(ddf(x), -df(x))

        # Steepest descent update
        xnew = x + s

        # Save optimized solution for plotting
        steps.append(xnew)
        
        fhist.append(f(xnew))

        # Check convergence
        
        if ( np.abs(fhist[-1] - fhist[-2]) < tol ):
            break

        x = xnew
        
    print('optimal solution is:', x)
        
    return steps, fhist, i 


# In[ ]:


# Initial guess
x0 = np.array([2, 2./5])
# Newton's method
steps, fhist, iterations = NewtonMethod(f1,df1,ddf1,x0,50,1e-6)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f1,steps=np.array(steps),fhist=np.array(fhist))


# In[ ]:


# Initial guess
x0 = np.array([-1, -1.0])
# Newton's method
steps, fhist, iterations = NewtonMethod(f2,df2,ddf2,x0,50,1e-6)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f2,steps=np.array(steps),fhist=np.array(fhist))


# In[ ]:


# Initial guess
x0 = np.array([-0.5, -1])
# Newton's method
steps, fhist, iterations = NewtonMethod(f3,df3,ddf3,x0,50,1e-8)
print('converged in', iterations, 'iterations')
# Plot convergence   
plotFunction(f3,steps=np.array(steps),levels=np.logspace(0,4,8), fhist=np.array(fhist))

plotConvergence( steps, [1 , 1] , 2 )

