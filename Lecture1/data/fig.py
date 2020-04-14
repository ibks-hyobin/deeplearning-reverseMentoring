#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# In[3]:


def show_3dgraph(first_X,first_Y,second_X,second_Y,third_X,third_Y):
    fig = plt.figure(figsize=(15,5))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(first_X[:, 0], first_X[:, 1], first_Y , c=first_Y, cmap='jet')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('y')
    ax1.set_zlim(-10, 6)
    ax1.view_init(40, -40)
    ax1.set_title('True train y')
    ax1.invert_xaxis()

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(second_X[:, 0], second_X[:, 1], second_Y, c=second_Y, cmap='jet')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('y')
    ax2.set_zlim(-10, 6)
    ax2.view_init(40, -40)
    ax2.set_title('Predicted train y')
    ax2.invert_xaxis()

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(third_X[:, 0], third_X[:, 1], third_Y, c=third_Y, cmap='jet')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('y')
    ax3.set_zlim(-10, 6)
    ax3.view_init(40, -40)
    ax3.set_title('Predicted validation y')
    ax3.invert_xaxis()

    plt.show()


# In[ ]:


def show_2dgragh(test_X, test_Y, pred_y):
    fig = plt.figure(figsize=(10,4))
    # ====== True Y Scattering ====== #
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(test_X[:, 0], test_X[:, 1], test_Y, c=test_Y, cmap='jet')

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('y')
    ax1.set_zlim(-10, 6)
    ax1.view_init(40, -40)
    ax1.set_title('True train y')
    ax1.invert_xaxis()

    # ====== Predicted Y Scattering ====== #
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(test_X[:, 0], test_X[:, 1], pred_y, c=pred_y, cmap='jet')

    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('y')
    ax2.set_zlim(-10, 6)
    ax2.view_init(40, -40)
    ax2.set_title('Predicted train y')
    ax2.invert_xaxis()


# In[ ]:




