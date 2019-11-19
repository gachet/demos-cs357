
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from io import StringIO
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
get_ipython().magic('matplotlib inline')


# In[2]:


params = ["radius", "texture", "perimeter", "area",
          "smoothness", "compactness", "concavity",
          "concave points", "symmetry", "fractal dimension"];
stats = ["(mean)", "(stderr)", "(worst)"]
labels = ["patient ID", "Malignant/Benign"]

for p in params:
    for s in stats:
        labels.append(p + " " + s)

tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat",header=None,names=labels)

features = tumor_data[labels[2:]]

features.head()


# In[3]:


mean_label = [labels[1]] + labels[2::3] 
print(mean_label)


# In[4]:


sns.pairplot(tumor_data[mean_label], hue="Malignant/Benign", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})


# In[5]:


corr_matrix = features.corr()
# plot correlation matrix
fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(corr_matrix, cmap=cm.get_cmap('jet'))
plt.title('Tumor features correlation matrix')
plt.grid('off')
ax1.set_xticks(np.arange(features.shape[1]))
ax1.set_yticks(np.arange(features.shape[1]))
ax1.set_xticklabels(labels[2:],fontsize=10,rotation=90)
ax1.set_yticklabels(labels[2:],fontsize=10)
fig.colorbar(cax)


# In[6]:


fig = plt.figure(figsize = (14,8))
ax1 = fig.add_subplot(111)
cax = ax1.imshow(tumor_data[mean_label[1:]].corr(), cmap=cm.get_cmap('jet'))
plt.title('Tumor features correlation matrix')
plt.grid('off')
ax1.set_xticks(np.arange(len(mean_label[1:])))
ax1.set_yticks(np.arange(len(mean_label[1:])))
ax1.set_xticklabels(mean_label[1:],fontsize=16,rotation=90)
ax1.set_yticklabels(mean_label[1:],fontsize=16)
fig.colorbar(cax)


# In[7]:


X = (features - features.mean())/features.std()

U, S, Vt = np.linalg.svd(X, full_matrices=False)

variances = S**2

V = Vt.T


# In[8]:


tot = sum(variances)
var_exp = [(i / tot)*100 for i in variances]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(len(var_exp)),var_exp, align='center', label='individual explained variance')
plt.step(range(len(var_exp)), cum_var_exp, 'r', where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(bbox_to_anchor=(1.1, 1.05))


# In[9]:


cum_var_exp


# In[10]:


k = 6
Xstar=X@V[:,:k]
tumor_pca_data = tumor_data[labels[:2]].copy()
pca_labels = []
for i in range(k):
    lab = 'p'+str(i)
    pca_labels += [lab]
    tumor_pca_data[lab] = Xstar[:,i]  


# In[11]:


sns.stripplot(x="p0",y="Malignant/Benign", data=tumor_pca_data, jitter=0.05, linewidth=1)


# In[12]:


def plot_arrow(vector,scale,text_label,text_posx,text_posy):
    plt.arrow(0, 0, scale*vector[0], scale*vector[1], head_width=0.1, head_length=0.1, fc='r', ec='r', lw=5)
    plt.text(scale*vector[0]*text_posx, scale*vector[1]*text_posy, text_label , color='black', ha='center', va='center', fontsize=18)


# In[13]:


g1 = sns.lmplot('p0', 'p1', tumor_pca_data, hue='Malignant/Benign', fit_reg=False, size=8, scatter_kws={'alpha':0.7,'s':60})

ax = g1.axes[0,0]
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')


# In[14]:


labelset = ['Malignant/Benign','p0', 'p1','p2','p3']
sns.pairplot(tumor_pca_data[labelset],hue='Malignant/Benign', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})


# In[15]:


plt.subplots(figsize = (14,8))
plt.bar(labels[2:],V[:,0])
plt.xticks(rotation=90)
plt.title('influence of original features in p0')

