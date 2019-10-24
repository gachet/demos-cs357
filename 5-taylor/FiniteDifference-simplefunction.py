
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set(font_scale=2)
sns.set_style("whitegrid")


# In[ ]:


def f(x):
    return np.exp(x) - 2

def df(x):
    return np.exp(x)

def df2(x):
    return np.exp(x)


# In[ ]:


x = np.linspace(-1, 2, 100)
plt.plot(x, f(x), lw=3)


# Let's evaluate the first derivative using finite difference approximation for decreasing values of h

# In[ ]:


xx = 1.0
h = 1.0
errors = []
hs = []

dfexact = df(xx) 


for i in range(20):
    dfapprox = (f(xx+h) - f(xx)) / h
    
    err = np.abs(dfexact - dfapprox)
    
    print(" %E \t %E " %(h, err) )
    
    hs.append(h)
    errors.append(err)
    
    h = h / 2


# In[ ]:


plt.loglog(hs, errors, lw=3)
plt.xlabel('h')
plt.ylabel('error')
plt.xlim([1e-6,1])
plt.ylim([1e-6,1])


# What happens if we keep decreasing the perturbation h?

# In[ ]:


xx = 1.0
h = 1.0
errors = []
hs = []

dfexact = df(xx) 
fxx = f(xx)
print('f exact = ',fxx)

for i in range(60):
    
    fxxh = f(xx+h)
    
    dfapprox = (fxxh - fxx) / h
    
    err = np.abs(dfexact - dfapprox)   
    
    print(" %E \t %E\t %E" %(h, fxxh-fxx, err) )
    hs.append(h)
    errors.append(err)
    
    h = h / 2


# In[ ]:


plt.loglog(hs, errors, lw=3)
plt.xlabel('h')
plt.ylabel('error')


# In[ ]:


plt.loglog(hs, errors, lw=3,label='total')
plt.loglog(hs,np.array(hs)*0.5*np.exp(1),'--',label='truncation')
plt.loglog(hs,2*2.2e-16/np.array(hs),'--',label='rounding')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel('h')
plt.ylabel('error')

