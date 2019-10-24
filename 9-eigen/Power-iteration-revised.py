
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Obtaining eigenvalues and eigenvectors numerically

# We want to prepare a matrix with deliberately chosen eigenvalues. Let's use diagonalization to write the matrix $\mathbf{A}$:
# 
# $$ \mathbf{A} = \mathbf{U D U}^{-1} $$
# 
# where we set ${\bf D}$ to be a known matrix with the pre-defined eigenvalues:
# 
# ```python
# D = np.diag([lambda1, lambda2, ..., lambdan])
# ```
# 
# We need to generate a matrix $\mathbf{U}$ that has an inverse. Orthogonal matrices are a great option here, since $\mathbf{U}^{-1} = \mathbf{U}^T$. We use QR decomposition to get an orthogonal matrix (you don't need to understand this method).

# In[ ]:


n = 4
X = np.random.rand(n,n)
U,_ = sla.qr(X)

D = np.diag([6,2,4,7])


# Now we can use diagonalization to write $\mathbf{A}$

# In[ ]:


A = U@D@U.T


# And we can check that the eigenvalues are indeed what we expected:

# In[ ]:


eigl, eigv = la.eig(A)
print(eigl)
print(eigv)


# We want to find the eigenvector corresponding to the largest eigenvalue in magnitude. For that, we can use `np.argsort`, which returns the indices that sort the array in ascending order. Hence, we are interested in the last entry.

# In[ ]:


eig_index_sort = np.argsort(abs(eigl))
print(eig_index_sort)
eigpos = eig_index_sort[-1]


# Recall that eigenvectors are stored as columns! Hence this would be the eigenvector corresponding to the largest (in magnitude) eigenvalue.

# In[ ]:


eigv[:,eigpos]


# Let's also pick an initial vector:

# In[ ]:


x0 = np.random.randn(n)
x0


# # Power Iteration
# Power iteration should converge to a multiple of the eigenvector ${\bf u}_1$ corresponding to largest eigenvalue (in magnitude).

# $$ {\bf x}_k = (\lambda_1)^k \left[ \alpha_1 {\bf u}_1 + \alpha_2 \left(\frac{\lambda_2}{\lambda_1}\right)^k{\bf u}_2 + ...  \right] $$

# Let's implememt power iteration. We simply perform multiple matrix vector multiplications using a for loop:

# In[ ]:


x = x0
for i in range(40):
    x = A @ x
    print(x)
    
print('Exact eigenvalue = ',eigv[:,eigpos])


# * What's the problem with this method?
# * Does anything useful come of this?
# * How do we fix it?

# We can get the corresponding eigenvalue

# In[ ]:


np.dot(x,A@x)/np.dot(x,x)


# # Normalized power iteration

# Back to the beginning: Reset to the initial vector and normalize

# In[ ]:


x = x0/la.norm(x0)


# Implement normalized power iteration. We will start with 10 iterations, and see what happens...

# In[ ]:


x = x0/la.norm(x0)

for i in range(10):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print(x)

print('exact = ' ,eigv[:,eigpos])

print('eig_approx = ', np.dot(x,A@x)/np.dot(x,x))


# ### What if the starting guess does not have any component of ${\bf u}_1$, i.e., if $\alpha_1 = 0$? 

# $$ {\bf x}_k = (\lambda_1)^k  \alpha_1 {\bf u}_1 + (\lambda_1)^k  \left(\frac{\lambda_2}{\lambda_1}\right)^k \alpha_2 {\bf u}_2 + (\lambda_1)^k \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \alpha_3{\bf u}_3 +  ...  \right] $$
# 
# In theory (or infinite precision calculations), if $\alpha_1=0$, power iteration will converge to a vector that is a multiple of the eigenvector ${\bf u}_2$. 
# 
# 
# In practice, it is unlikely that a random vector ${\bf x}_0$ will not have any component of ${\bf u}_1$. In the chances that happens, finite operations during the iterative process will usually introduce such component.

# ### Creating a matrix where the dominant eigenvalue is negative

# In[ ]:


n = 4
    
D = np.diag([-5,2,4,3])

A = U@D@U.T

eigl, eigv = la.eig(A)

eig_index_sort = np.argsort(abs(eigl))
eigpos = eig_index_sort[-1]

print(eigl)
print(eigv[:,eigpos])


# In[ ]:


x = x0/la.norm(x0)

for i in range(40):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print(x)

print('exact = ' ,eigv[:,eigpos])

print('eig_approx = ', np.dot(x,A@x)/np.dot(x,x))

print(D)


# What is happening here? Note that the scalar that multiplies the eigenvector ${\bf u}_1$ in 
# 
# $$ {\bf x}_k = (\lambda_1)^k  \alpha_1 {\bf u}_1 + (\lambda_1)^k  \left(\frac{\lambda_2}{\lambda_1}\right)^k \alpha_2 {\bf u}_2 + (\lambda_1)^k \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \alpha_3{\bf u}_3 +  ...  \right] $$
# 
# is $(\lambda_1)^k$, and hence if the eigenvalue  $\lambda_1$ is negative, the solution of power iteration will converge to the eigenvector, but with alternating signs, i.e., ${\bf u}_1$ and $-{\bf u}_1$.

# ### When dealing with dominant eigenvalues of multiplicity greater than 1: $|\lambda_1| = |\lambda_2| $ and $ \lambda_1, \lambda_2 > 0 $
# 

# In[ ]:


n = 4

D = np.diag([5,5,2,1])

A = U@D@U.T

eigl, eigv = la.eig(A)

print(eigl)
print(eigv[:,2])
print(eigv[:,3])


# In[ ]:


x = x0/la.norm(x0)

for i in range(40):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print(x)

print('u1_exact = ' ,eigv[:,2])
print('u2_exact = ' ,eigv[:,3])

print('eig_approx = ', np.dot(x,A@x)/np.dot(x,x))

print(D)


# In general, power method converges to:
# 
# $$ {\bf x}_k = (\lambda_1)^k  \alpha_1 {\bf u}_1 + (\lambda_1)^k  \left(\frac{\lambda_2}{\lambda_1}\right)^k \alpha_2 {\bf u}_2 + (\lambda_1)^k \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \alpha_3{\bf u}_3 +  ...  \right] $$
# 
# However if $|\lambda_1| = |\lambda_2| $ and $ \lambda_1, \lambda_2 > 0 $, we get:
# 
# $$ {\bf x}_k = (\lambda_1)^k \left( \alpha_1 {\bf u}_1 + \alpha_2 {\bf u}_2 \right) + \left[ ...  \right] $$
# 
# and hence the solution of power iteration will converge to a multiple of the linear combination of the eigenvectors ${\bf u}_1$ and ${\bf u}_2$.

# ### When dealing with dominant eigenvalues of multiplicity greater than 1: $|\lambda_1| = |\lambda_2| $ and $ \lambda_1, \lambda_2 < 0 $
# 

# In[ ]:


n = 4

D = np.diag([-5,-5,2,1])

A = U@D@U.T

eigl, eigv = la.eig(A)

print(eigl)
print(eigv[:,2])
print(eigv[:,3])


# In[ ]:


x = x0/la.norm(x0)

for i in range(40):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print(x)

print('u1_exact = ' ,eigv[:,2])
print('u2_exact = ' ,eigv[:,3])

print('eig_approx = ', np.dot(x,A@x)/np.dot(x,x))

print(D)


# In general, power method converges to:
# 
# $$ {\bf x}_k = (\lambda_1)^k  \alpha_1 {\bf u}_1 + (\lambda_1)^k  \left(\frac{\lambda_2}{\lambda_1}\right)^k \alpha_2 {\bf u}_2 + (\lambda_1)^k \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \alpha_3{\bf u}_3 +  ...  \right] $$
# 
# However if $|\lambda_1| = |\lambda_2| $ and $ \lambda_1, \lambda_2 < 0 $, we get:
# 
# $$ {\bf x}_k = \pm |\lambda_1|^k \left( \alpha_1 {\bf u}_1 + \alpha_2 {\bf u}_2 \right) + \left[ ...  \right] $$
# 
# and hence the solution of power iteration will converge to a multiple of the linear combination of the eigenvectors ${\bf u}_1$ and ${\bf u}_2$, but the signs will flip at each step of the iterative method.

# ### When dealing with dominant eigenvalues of multiplicity greater than 1: $|\lambda_1| = |\lambda_2| $ and $ \lambda_1 , \lambda_2 $ have opposite signs
# 

# In[ ]:


n = 4

D = np.diag([-5,5,2,1])

A = U@D@U.T

eigl, eigv = la.eig(A)

print(eigl)
print(eigv[:,0])
print(eigv[:,1])


# In[ ]:


x = x0/la.norm(x0)

for i in range(40):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print(x)

print('u1_exact = ' ,eigv[:,2])
print('u2_exact = ' ,eigv[:,3])

print('eig_approx = ', np.dot(x,A@x)/np.dot(x,x))

print(D)


# In general, power method converges to:
# 
# $$ {\bf x}_k = (\lambda_1)^k  \alpha_1 {\bf u}_1 + (\lambda_1)^k  \left(\frac{\lambda_2}{\lambda_1}\right)^k \alpha_2 {\bf u}_2 + (\lambda_1)^k \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \alpha_3{\bf u}_3 +  ...  \right] $$
# 
# However if $|\lambda_1| = |\lambda_2| $, $ \lambda_1, \lambda_2$ have opposite signs, we get:
# 
# $$ {\bf x}_k = \pm |\lambda_1|^k \left( \alpha_1 {\bf u}_1 \pm \alpha_2 {\bf u}_2 \right) + \left[ ...  \right] $$
# 
# and hence power iteration does not converge to one solution. Indeed, the method oscilates between two linear combination of eigenvectors, and fails to give the correct eigenvalue. 

# ### Summary - Pitfalls of power iteration:
# 
# - Risk of eventual overflow. Use normalized power iteration to avoid this.
# 
# 
# - If the initial guess has $\alpha_1 = 0$, the method will converge to multiple of eigenvector ${\bf u}_2$ if infinite precision computation is used. In practice (in finite precision computations), this will not be an issue, and the method will converge to multiple of eigenvector ${\bf u}_1$.
# 
# 
# - If the two largest eigenvalues are equal in magnitude, power iteration will converge to a vector that is a linear combination of the corresponding eigenvectors (or fail to converge). This is a real problem that cannot be discounted in practice. Other methods should be used in this case.
# 

# # Estimating the eigenvalue

# We want to approximate the eigenvalue ${\bf u}_1$ using the solution of power iteration
# 
# $$ {\bf x}_k = (\lambda_1)^k  \alpha_1 {\bf u}_1 + (\lambda_1)^k  \left(\frac{\lambda_2}{\lambda_1}\right)^k \alpha_2 {\bf u}_2 + (\lambda_1)^k \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \alpha_3{\bf u}_3 +  ...  \right] $$
# 
# 
# $ {\bf x}_k $ approaches a multiple of the eigenvector ${\bf u}_1$ as $k \rightarrow \infty$, hence
# 
# $$ {\bf x}_k  =   (\lambda_1)^k  \alpha_1 {\bf u}_1 $$
# 
# but also  
# 
# $$ {\bf x}_{k+1}  =   (\lambda_1)^{k+1}  \alpha_1 {\bf u}_1 \Longrightarrow {\bf x}_{k+1} = \lambda_1 {\bf x}_{k} $$
# 
# We can then approximate $\lambda_1$ as the ratio of corresponding entries of the vectors ${\bf x}_{k+1}$ and ${\bf x}_{k}$, i.e., 
# 
# $$ \lambda_1 \approx \frac{({\bf x}_{k+1})_j } { ({\bf x}_{k})_j }$$
# 

# # Error of Power Iteration

# We define the approximated eigenvector as 
# 
# $$ {\bf u}_{approx} = \frac{{\bf x}_k } { (\lambda_1)^k  \alpha_1} $$
# 
# and hence the error becomes the part of the power iteration solution that was neglected, i.e.,
# 
# $$ {\bf e} =  {\bf u}_{approx} - {\bf u}_1 = \left(\frac{\lambda_2}{\lambda_1}\right)^k \frac{\alpha_2}{\alpha_1} {\bf u}_2 +  \left[   \left(\frac{\lambda_3}{\lambda_1}\right)^k \frac{\alpha_3}{\alpha_1}{\bf u}_3 +  ...  \right]  $$
# 
# and when $k$ is large, we can write (again, we are assuming that $|\lambda_1| > |\lambda_2|  \ge |\lambda_3|  \ge |\lambda_4| ... $ 
# 
# $${\bf e}_k \approx \left(\frac{\lambda_2}{\lambda_1}\right)^k \frac{\alpha_2}{\alpha_1} {\bf u}_2 $$
# 
# And when we take the norm of the error
# 
# $$||{\bf e}_k|| \approx \left|\frac{\lambda_2}{\lambda_1}\right|^k \left|\frac{\alpha_2}{\alpha_1}\right| ||{\bf u}_2 || \rightarrow ||{\bf e}_k|| = O\left(\left|\frac{\lambda_2}{\lambda_1}\right|^k \right)$$

# # Convergence of Power Iteration
# 
# We want to see what happens to the error from one iteration of the other of power iteration
# 
# $$ \frac{||{\bf e}_{k+1}||}{||{\bf e}_{k}||} = 
# \frac{\left|\frac{\lambda_2}{\lambda_1}\right|^{k+1} \left|\frac{\alpha_2}{\alpha_1}\right|  }{\left|\frac{\lambda_2}{\lambda_1}\right|^k \left|\frac{\alpha_2}{\alpha_1}\right| } = \frac{\lambda_2}{\lambda_1} $$ 
# 
# Or in other words, we can say that the error decreases by a **constant** value, given as $\frac{\lambda_2}{\lambda_1} $, at each iteration.
# 
# ** Power method has LINEAR convergence! **
# 
# $$ ||{\bf e}_{k+1}|| = \frac{\lambda_2}{\lambda_1} ||{\bf e}_{k}||$$   
# or we can also write $$ ||{\bf e}_{k+1}|| = \left(\frac{\lambda_2}{\lambda_1} \right)^k||{\bf e}_{0}||$$

# # Simple Example:
# Suppose you are given a matrix with eigenvalues:
# 
# $$[3,4,5]$$
# 
# You use normalized power iteration to approximate one of the eigenvectors, which is given as ${\bf x}$, and we assume $||{\bf x} || = 1$.
# 
# You knew the norm of the error of the initial guess was given as
# 
# $$|| {\bf e}_0 || = ||{\bf x} - {\bf x}_0 || = 0.3 $$
# 
# How big will be the error after three rounds of power iteration? (Since all vectors have norm 1, the absolute and relative error are the same)
# 
# 
# $$|| {\bf e}_3 || = \left| \frac{4}{5} \right|^3 || {\bf e}_0 || = 0.3 \left| \frac{4}{5} \right|^3  = 0.1536 $$

# # Convergence plots

# In[ ]:


n=4

lambda_array_ordered = [7, 3, -2, 1]

X = np.random.rand(n,n)
U,_ = sla.qr(X)
D = np.diag(lambda_array_ordered)
A = U@D@U.T
eigl, eigv = la.eig(A)

eig_index_sort = np.argsort(abs(eigl))
eigpos = eig_index_sort[-1]
u1_exact = eigv[:,eigpos]

print('Largest lambda = ', lambda_array_ordered[0])
print('Eigenvector = ', u1_exact)
print('Convergence rate = ', lambda_array_ordered[1]/lambda_array_ordered[0])


# In[ ]:


# Generate normalized initial guess
x0 = np.random.random(n)
x = x0/la.norm(x0)

count = 0
diff  = 1
eigs  = [x[0]]
error = [np.abs( eigs[-1]  - lambda_array_ordered[0] )]

# We will use as stopping criteria the change in the
# approximation for the eigenvalue

while (diff > 1e-6 and count < 100):
    count += 1
    xnew = A@x #xk+1 = A xk
    eigs.append(xnew[0]/x[0])
    x = xnew/la.norm(xnew)    
    diff  = np.abs( eigs[-1]  - eigs[-2] )
    error.append( np.abs( eigs[-1]  - lambda_array_ordered[0] ) )    
    print("% 10f % 2e % 2f" %(eigs[-1], error[-1], error[-1]/error[-2])) 


# In[ ]:


plt.semilogy(np.abs(error)) 


# # Inverse Power iteration
# 
# What if we are interested in the smaller eigenvalue in magnitude?
# 
# Suppose ${\bf x},\lambda$ is an eigenpair of ${\bf A}$, such that ${\bf A}{\bf x}  = \lambda {\bf x}$. What would be an eigenvalue of  ${\bf A}^{-1}$?
# 
# 
# $${\bf A}^{-1}{\bf A}{\bf x}  = {\bf A}^{-1}\lambda {\bf x}$$
# 
# $${\bf I}{\bf x}  =  \lambda {\bf A}^{-1} {\bf x}$$
# 
# $$\frac{1}{\lambda}{\bf x}  =  {\bf A}^{-1} {\bf x}$$
# 
# 
# ** Hence $\frac{1}{\lambda}$ is an eigenvalue of ${\bf A}^{-1} $ **.
# 
# If we want to find the smallest eigenvalue in magnitude of ${\bf A}$, we can perform power iteration using the matrix ${\bf A}^{-1}$ to find $\bar\lambda = \frac{1}{\lambda}$, where  $\bar\lambda$ is the largest eigenvalue of ${\bf A}^{-1}$.
# 
# Let's implement that:

# In[ ]:


n = 4
D = np.diag([5,-1,2,7])

A = U@D@U.T
eigl, eigv = la.eig(A)

eig_index_sort = np.argsort(abs(eigl))
eig_index_sort
eigpos = eig_index_sort[0]

print(eigv[:,eigpos])


# In[ ]:


x0 = np.random.random(n)
nrm = la.norm(x0)
x = x0/nrm

for i in range(20):
    x = la.inv(A)@x
    x = x/la.norm(x)

print("lambdan = ",x.T@A@x/(x.T@x))
print("un = ", x) 


# Can you find ways to improve the code snippet above? 

# In[ ]:


#Inverse Power iteration to get smallest eigenvalue
x0 = np.random.random(n)
nrm = la.norm(x0)
x = x0/nrm
P, L, Um = sla.lu(A)
for k in range(20):
    y = sla.solve_triangular(L, np.dot(P.T, x), lower=True)
    x = sla.solve_triangular(Um, y)
    x = x/la.norm(x)

print("lambdan = ",x.T@A@x/(x.T@x))
print("un = ", x)  


# # Inverse Shifted Power Iteration
# 
# What if we want to find another eigenvalue that is not the largest or the smallest? 
# 
# Suppose ${\bf x},\lambda$ is an eigenpair of ${\bf A}$, such that ${\bf A}{\bf x}  = \lambda {\bf x}$. We want to find the eigenvalues of the shifted inverse matrix $({\bf A} - \sigma{\bf I})^{-1}$
# 
# 
# $$({\bf A} - \sigma{\bf I})^{-1}{\bf x}  = \bar\lambda {\bf x}$$
# 
# $${\bf I}{\bf x}  =  \bar\lambda ({\bf A} - \sigma{\bf I}) {\bf x} = \bar\lambda ({\lambda \bf I} - \sigma{\bf I}) {\bf x}$$
# 
# $$  \bar\lambda  = \frac{1}{\lambda-\sigma}$$
# 
# 
# We could write the above eigenvalue problem as 
# 
# 
# $$ {\bf B}^{-1}{\bf x}  = \bar\lambda {\bf x}$$
# 
# which can be solved by inverse power iteration, which will converge to the eigenvalue $\frac{1}{\lambda-\sigma}$

# In[ ]:


n = 4
D = np.diag([5,-7,2,10])

A = U@D@U.T
eigl, eigv = la.eig(A)

print(eigl)
eigv


# In[ ]:


#Shifted Inverse Power iteration 
sigma = 1

x0 = np.random.random(n)
nrm = la.norm(x0)
x = x0/nrm
B = A - sigma*np.eye(n)
P, L, Um = sla.lu(B)
for k in range(20):
    y = sla.solve_triangular(L, np.dot(P.T, x), lower=True)
    x = sla.solve_triangular(Um, y)
    x = x/la.norm(x)

print("lambdan = ",x.T@A@x/(x.T@x))
print("un = ", x)  


# # Computational cost and convergence
# 
# - power iteration: to obtain largest eigenvalue in magnitude ${\lambda_1}$
#     - Matrix-vector multiplications at each iteration: $O(n^2)$
#     - convergence rate: $\left|\frac{\lambda_2}{\lambda_1} \right|$
#     
#    
# - inverse power iteration: to obtain smallest eigenvalue in magnitude ${\lambda_n}$
#     - only one factorization: $O(n^3)$
#     - backward-forward substitutions to solve at each iteration: $O(n^2)$
#     - convergence rate: $\left|\frac{\lambda_n}{\lambda_{n-1}} \right|$  
#     
#    
# - inverse shifted power iteration: to obtain an eigenvalue close to a known/given value $\sigma$
#     - only one factorization: $O(n^3)$
#     - backward-forward substitutions to solve at each iteration: $O(n^2)$
#     - convergence rate: $\left|\frac{\lambda_c - \sigma}{\lambda_{c2} - \sigma} \right|$  
#     where $\lambda_c$ is the closest eigenvalue to $\sigma$ and $\lambda_{c2}$ is the second closest eigenvalue to $\sigma$.
#    
