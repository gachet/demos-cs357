
# coding: utf-8

# # 2D Elliptic problem (sparse)

# In[ ]:


import scipy as sparse
import scipy.sparse.linalg as spla
N = 12
A,X,Y = FDM_system(N)
print(type(A))


# In[ ]:


plt.spy(A,markersize = 1)


# In[ ]:


def f(x,y):
    c = 1 + 8*np.pi**2
    return c*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

b = FDM_rhs(f,X,Y)


# Use Conjugate Gradient Method (an iterative solution method) to solve this equation

# In[ ]:


u,info = spla.cg(A,b)
U = solution_to_2D(u)
plt.contourf(X,Y,U,cmap ='coolwarm')
plt.colorbar()

