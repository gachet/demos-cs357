
# coding: utf-8

# ## Iclicker question

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from math import factorial


# In[ ]:


import sympy as sp
sp.init_printing()
x = sp.Symbol("x")


# In[ ]:


f = sp.exp(x)

f


# In[ ]:


def plot_sympy(my_f, my_pts, **kwargs):
    f_values = np.array([my_f.subs(x, pt) for pt in my_pts])
    plt.plot(pts, f_values, **kwargs)

    
def semilogy_sympy(my_f, my_pts, **kwargs):
    f_values = np.array([my_f.subs(x, pt) for pt in my_pts])
    plt.semilogy(pts, f_values, **kwargs)


# In[ ]:


n = 4
xo = 2


# In[ ]:


taylor = 0

for i in range(n):
    taylor += f.diff(x, i).subs(x, xo)/factorial(i) * (x-xo)**i

error =  f - taylor


# In[ ]:


pts = np.linspace(-1, 4, 100)
plot_sympy(taylor, pts, label="taylor n=3")
plot_sympy(f, pts, label="f")
plt.legend(loc="best")
plt.grid()
plt.xlabel('$x$')
plt.ylabel('function values')


# In[ ]:


semilogy_sympy(error, pts, label="error")
f2=x**2
f3=x**3
f4=x**4
f5=x**5
semilogy_sympy(f2, pts, label="$x^2$")
plot_sympy(f3, pts, label="$x^3$")
plot_sympy(f4, pts, label="$x^4$")
plot_sympy(f5, pts, label="$x^5$")
plt.legend(loc="best")
plt.grid()
plt.xlabel('$x$')
plt.ylabel('error')


# In[ ]:


semilogy_sympy(error, pts, label="error")
f2=abs((x-2)**2)
f3=abs((x-2)**3)
f4=abs((x-2)**4)
f5=abs((x-2)**5)
semilogy_sympy(f2, pts, label="$(x-2)$")
semilogy_sympy(f3, pts, label="$(x-2)^3$")
semilogy_sympy(f4, pts, label="$(x-2)^4$")
semilogy_sympy(f5, pts, label="$(x-2)^5$")
plt.legend(loc="best")
plt.grid()
plt.xlabel('$x$')
plt.ylabel('error')
plt.xlim([1.5,2.5])

