
# coding: utf-8

# In[9]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("whitegrid")


# # One dimensional nonlinear equations

# In this activity, we will find the root of nonlinear equations using three different iterative methods. For each one, you should be thinking about cost and convergence rate.
# 
# The iterative methods below can be applied to more complex equations, but here we will use a simple nonlinear equation of the form:
# 
# $$f(x) = e^x - 2 $$
# 
# The exact root that satisfies $f(x) = 0$ is $x = ln(2) \approx 0.693147$. We can take a look at the function in the interval $[-2,2]$.
# 

# In[63]:


a0 = -2
b0 = 2

x = np.linspace(a0, b0)

def f(x):
    return np.exp(x) - 2

def df(x):
    return np.exp(x)

xtrue = np.log(2)

plt.plot(x, f(x))
plt.plot(xtrue,0,'ro')


# ## Bisection Method
# 
# #### First we will run the iterative process for a few iterations:

# In[52]:


a = a0
b = b0
interval = np.abs(a - b)   
errors = []

fa = f(a)

for i in range(12):
    m = (a+b)/2
    fm = f(m)   
    if  np.sign(fa) == np.sign(fm):
        a = m 
        fa = fm # this line is not really needed, 
        # since we only need the sign of a, and sign of a is the same as sign of m
    else:
        b = m
    interval = np.abs(a - b)    
    errors.append(interval)        
    print("%10g \t %10g \t %12g" % (a, b, interval))
    
print('exact root is = ', np.log(2))


# #### Now we will add a stopping criteria. 
# 
# Since we know the interval gets divided by two every iteration, how many iterations do we need to perform to achieve the tolerance $2^{-k}$?
# 
# Note that only one function evaluation is needed per iteration!
# 
# We can also verify the linear convergence, with C = 0.5

# In[53]:


a = a0
b = b0
interval = np.abs(a - b)   
errors = [interval]

fa = f(a)
count = 0

while count < 30 and interval > 2**(-4):
    m = (a+b)/2
    fm = f(m)   
    if  fa*fm > 0:
        a = m 
    else:
        b = m
    interval = np.abs(a - b)    
    errors.append(interval)        
    print("%10g \t %10g \t %12g %12g" % (a, b, interval, interval/errors[-2]))
    
print('exact root is = ', np.log(2))


# In[58]:


plt.plot(errors)
plt.ylabel('Error (interval size)')
plt.xlabel('Iterations')


# What happens if you have multiple roots inside the interval? Bisection method will converge to one of the roots. Try to run the code snippet above using the function 
# 
# $$ f(x) = 0.5 x^2 - 2 $$
# 
# Change the interval, and observe what happens.

# ## Newton's Method

# In[98]:


x0 = 2
r = 2


# In[99]:


x = x0
count = 0
tol = 1e-6
err = 1
errors = [err]

while count < 30 and err > tol:
    x = x - f(x)/df(x)
    err = abs(x-xtrue)
    errors.append(err)
    print('%10g \t%10g \t %.16e %.4g' % (x, f(x), err, errors[-1]/(errors[-2]**r) ))


# ## Secant Method

# In[94]:


x0 = 2
x1 = 8
r = 2
#r = 1.618


# In[95]:


# Need two initial guesses
xbefore = x0
x = x1
count = 0
tol = 1e-8
err = 1
errors = [err]

while count < 30 and err > tol:

    df_approx = (f(x)-f(xbefore))/(x-xbefore)
    xbefore = x
    x = x - f(x)/df_approx
    err = abs(x-xtrue)
    errors.append(err)
    print('%10g \t%10g \t %.16e %.4g' % (x, f(x), err, errors[-1]/errors[-2]**r ))


# # N-Dimensional Nonlinear Equations

# We will solve the following system of nonlinear equations:
# 
# $$ x + 2y = 2 $$
# 
# $$ x^2 + 4y^2 = 4 $$
# 
# We will define our vector valued function ${\bf f}$, which takes a vector as argument, with the components $x$ and $y$. We are trying to find the numerical (approximated) solution that safisfies:
# 
# $${\bf f} = \begin{bmatrix} f_1 \\ f_2 \end{bmatrix}
#           = \begin{bmatrix} x + 2y - 2 \\ x^2 + 4y^2 - 4 \end{bmatrix}
#           = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
#           
# and the exact solution is $[0,1]$
# 
# We will also define the gradient of ${\bf f}$, $\nabla{\bf f}$, which we call the Jacobian.

# ### Newton's method

# In[11]:


def f(xvec):
    x, y = xvec
    return np.array([
        x + 2*y -2,
        x**2 + 4*y**2 - 4
        ])

def Jf(xvec):
    x, y = xvec
    return np.array([
        [1, 2],
        [2*x, 8*y]
        ])


# Pick an initial guess.

# In[15]:


x = np.array([1, 2])


# Now implement Newton's method.

# In[16]:


for i in range(10):
    s = la.solve(Jf(x), -f(x))
    x = x + s
x


# Check if that's really a solution:

# In[17]:


f(x)


# The cost is $O(n^3)$ per iteration, since the Jacobian changes every iteration. But how fast is the convergence?

# In[31]:


r = 2

xtrue = np.array([0, 1])
x = np.array([1, 2])
errors = [la.norm(x)]

while errors[-1] > 1e-12:
    A = Jf(x)
    s = la.solve(A, f(x))
    x = x - s
    err = la.norm(x-xtrue)
    errors.append(err)
    print(' %.16e \t %.4g' % (err, errors[-1]/errors[-2]**r ))


# ### Finite Difference

# Suppose you don't know how to calculate the Jacobian. You can use Forward Finite Difference to approximate the derivatives! 

# In[152]:


def fd(xvec,dx):
    fval = f(xvec)
    J = np.zeros((fval.shape[0],xvec.shape[0]))
    for j in range(len(xvec)):
        xvec[j] = xvec[j] + dx
        dfd = f(xvec)-fval
        for i in range(len(fval)):
            J[i,j] = dfd[i]/dx
        xvec[j] = xvec[j] - dx
    return J


# In[155]:


x = np.array([4, 2],dtype=float)
fd(x,0.0001)


# In[156]:


Jf(np.array([4,2]))

