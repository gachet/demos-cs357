
# coding: utf-8

# # Floating point vs Program Logic

# What will the following code snippet do?

# In[ ]:


from time import sleep

x = 0.0

while x != (1.0):
    x += 0.1
    print(repr(x))
    
    sleep(0.4)

