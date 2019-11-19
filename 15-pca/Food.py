
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
get_ipython().magic('matplotlib inline')

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd


# Example from:
# http://setosa.io/ev/principal-component-analysis/

# In[2]:


data = pd.io.parsers.read_csv("UK_foods.csv")
data.head()


# In[3]:


headers = data['Unnamed: 0'].values.tolist()
print(headers)

new_data = data.drop(['Unnamed: 0'], axis=1)
new_data.head()

regions = new_data.columns.values.tolist()
print(regions)


# In[4]:


food = pd.DataFrame(new_data.values.T,columns=headers)
food['region'] = regions
food


# ### This is when we want to try PCA!

# In[5]:


#Performing PCA without scaling the data (to match the results from the website)
X = pd.DataFrame(food[headers], columns=headers)
# In general, PCA scales the variables to zero-mean (use line below to scale)
# X = pd.DataFrame(scale(food[headers]), columns=headers)


# In[6]:


pca = PCA().fit(X)
pca_samples = pca.transform(X)


# In[7]:


var_exp = pca.explained_variance_ratio_
plt.bar(range(len(var_exp)),var_exp, align='center', label='individual explained variance');
plt.ylabel('Explained variance ratio');
plt.xlabel('Principal components');


# In[8]:


components = pd.DataFrame(pca.components_, columns = headers) 
components


# In[9]:


plt.figure()
plt.bar(headers,components.values[0])
plt.xticks(rotation=90)
plt.title('influence of original variables(food) upon pc1')
plt.figure()
plt.bar(headers,components.values[1])
plt.xticks(rotation=90)
plt.title('influence of original variables(food) upon pc2')


# In[10]:


Xstar = pd.DataFrame(pca_samples,columns=['pc1','pc2','pc3','pc4'])
Xstar['region'] = regions
Xstar


# In[11]:


sns.stripplot(x="pc1",y="region", data=Xstar, jitter=0.05, linewidth=1)


# In[12]:


ax = plt.figure()
ax = sns.lmplot('pc1', 'pc2',Xstar,hue='region', fit_reg=False)
plt.axis('equal')
plt.xlabel('pc1')
plt.ylabel('pc2')


# In[13]:


def plot_arrow(v,scale,text_pos,text_label):
    plt.arrow(0, 0, scale*v[0], scale*v[1], head_width=0.2, head_length=0.2, linewidth=2, color='red')
    plt.text(v[0]*text_pos, v[1]*text_pos, text_label, color='black', ha='center', va='center', fontsize=18)


# In[14]:


ax = plt.figure()
ax = sns.lmplot('pc1', 'pc2',Xstar,hue='region', fit_reg=False)
plt.axis('equal')
plt.xlabel('pc1')
plt.ylabel('pc2')

lab = ['Fresh_potatoes ',
 'Fresh_fruit ',
 'Soft_drinks ',
 'Alcoholic_drinks ']

for i,label in enumerate(lab):
    plot_arrow(components[label],500,500,label)

