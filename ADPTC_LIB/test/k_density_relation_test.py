'''
Description: 测试 密度 与  k 个数的关系
Author: SongJ
Date: 2020-12-29 11:07:22
LastEditTime: 2020-12-29 13:15:20
LastEditors: SongJ
'''
#%%
import numpy as np
import sys
sys.path.insert(0,r"D:\workspace_clustering\ATDPC_Package\ATDPC")
from atdpc import ATDPC
import myutil

# X = np.loadtxt(r"../test_data/Aggregation.txt", delimiter="\t")
# X = X[:,[0,1]]
# 
X = np.loadtxt(r"../test_data/gaussian_point.txt", delimiter=" ")
print(X)
# %%
atdpc_obj = ATDPC(X)
atdpc_obj.clustering(0.03,density_metric='gauss',knn_num=50,connect_eps=0.5,leaf_size=300,fast=False)
print(atdpc_obj.labels)
myutil.show_result(atdpc_obj.labels,X,np.array(list(atdpc_obj.core_points)))
atdpc_obj.calc_time
#%%
sorted_rel = np.sort(atdpc_obj.density_and_k_relation,axis=0)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(sorted_rel[:,1],sorted_rel[:,0])
# %%
k_beyond_100 = sorted_rel[np.where(sorted_rel[:,0]>204)]
fig, ax = plt.subplots()
ax.plot(k_beyond_100[:,1],k_beyond_100[:,0])
# %%
sorted_rel
# %%
atdpc_obj.density
# %%
np.where(atdpc_obj.density>204)[0].shape
# %%
atdpc_obj.density.shape
# %%
np.sort(atdpc_obj.density)[700:]
# %%
