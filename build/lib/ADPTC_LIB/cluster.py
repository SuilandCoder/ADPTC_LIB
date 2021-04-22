'''
Description:
Author: SongJ
Date: 2020-12-28 10:23:45
LastEditTime: 2021-04-10 10:51:23
LastEditors: SongJ
'''
#%%
from sklearn.utils import check_array
import time
from . import myutil
from . import visual
from . import prepare

class ADPTC:

    def __init__(self, X, lon_index=0, lat_index=1, time_index=2, attrs_index=[3]):
      self.X = X
      self.lon_index = lon_index
      self.lat_index = lat_index
      self.time_index = time_index
      self.attrs_index = attrs_index
      pass


    def clustering(self,eps, density_metric='cutoff', dist_metric='euclidean', algorithm='auto', knn_num=20, leaf_size=300, connect_eps=1,fast=False):
        '''
            description: 普通聚类，不考虑属性类型，计算混合距离
            return {*}
            eps：
                阈值
            density_metric:
                密度计算方式，默认为截断密度，支持 gauss
            dist_metric：
                距离计算方法，默认为 euclidean，支持['euclidean','braycurtis', 'canberra', 'chebyshev', 'cityblock',
                'correlation', 'cosine', 'dice',  'hamming','jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
                'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.]
            algorithm：
                近邻计算方法，计算最近邻点及距离，默认'kd_tree',支持['kd_tree','ball_tree']
            knn_num:
                近邻个数
            leaf_size:
                近邻计算方法中用到的叶子节点个数，会影响计算和查询速度
            connect_eps:
                密度连通性阈值
            fast:
                是否启用快速聚类，通过最近邻查找算法，优化斥群值查找速度
        '''
        start=time.clock()
        try:
            data = check_array(self.X, accept_sparse='csr')
        except:
            raise ValueError("输入的数据集必须为矩阵")
        dist_mat = myutil.calc_dist_matrix(data,dist_metric)
        density = myutil.calc_density(dist_mat,eps,density_metric)
        denser_pos,denser_dist,density_and_k_relation = myutil.calc_repulsive_force(data,density,knn_num,leaf_size,dist_mat,fast)
        if(-1 in denser_pos):
            raise ValueError('阈值太小啦~,或者尝试使用高斯密度呢：density_metric=gauss')
            pass
        gamma = myutil.calc_gamma(density,denser_dist)
        labels, core_points=myutil.extract_cluster_auto(data,density,eps,connect_eps,denser_dist,denser_pos,gamma,dist_mat)
        self.labels = labels
        self.core_points = core_points
        self.density_and_k_relation = density_and_k_relation
        self.density = density
        end=time.clock()
        self.calc_time = str(end-start)
        return self


    def spacial_clustering_raster(self,spatial_eps,attr_eps,density_metric='cutoff',dist_metric='euclidean', algorithm='auto', knn_num=20, leaf_size=300, connect_eps=1):
        '''
            description: 地理空间栅格数据聚类分析；输入数据为二维矩阵，行和列分别为地理空间的纬度和经度。
            return {*}
            spatial_eps：
                空间阈值
            attr_eps：
                属性阈值
            density_metric:
                密度计算方式，默认为截断密度，支持 gauss
            dist_metric：
                距离计算方法，默认为 euclidean，支持['euclidean','braycurtis', 'canberra', 'chebyshev', 'cityblock',
                'correlation', 'cosine', 'dice',  'hamming','jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
                'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.]
            algorithm：
                近邻计算方法，计算最近邻点及距离，默认'kd_tree',支持['kd_tree','ball_tree','brute','Annoy','HNSW']
            knn_num:
                近邻个数
            leaf_size:
                近邻计算方法中用到的叶子节点个数，会影响计算和查询速度
            connect_eps:
                密度连通性阈值
        '''
        start=time.clock()
        try:
            data = check_array(self.X)
        except:
            raise ValueError("Two-dimensional matrix is needed,the rows and columns are the latitude and longitude.")
        #* 立方体数据转二维矩阵，行为样本，列为要素
        data_not_none,pos_not_none = myutil.rasterArray_to_sampleArray(data)
        self.data_not_none = data_not_none
        mixin_near_matrix = myutil.calc_homo_near_grid(data,spatial_eps,attr_eps,pos_not_none)
        density = myutil.calc_gaus_density_spatial(data,spatial_eps,attr_eps)
        denser_pos,denser_dist,density_and_k_relation = myutil.calc_repulsive_force(data_not_none,density,knn_num,leaf_size,fast=True)
        if(-1 in denser_pos):
            raise ValueError('阈值太小啦~,或者尝试使用高斯密度呢：density_metric=gauss')
            pass
        gamma = myutil.calc_gamma(density,denser_dist)
        labels, core_points = myutil.extract_cluster_auto_st(data_not_none,density,connect_eps,denser_dist,denser_pos,gamma,mixin_near_matrix)
        self.labels = labels
        self.core_points = core_points
        self.density_and_k_relation = density_and_k_relation
        self.density = density
        end=time.clock()
        self.calc_time = str(end-start)
        return self


    def st_clustering_raster(self,spatial_eps,time_eps,attr_eps,density_metric='cutoff',dist_metric='euclidean', algorithm='auto', knn_num=20, leaf_size=300, connect_eps=1):
        '''
            description: 地理时空栅格数聚类，输入数据为三维矩阵，三个维度分别为纬度、经度、时间。
            return {*}
            spatial_eps：
                空间阈值
            time_eps:
                时间阈值
            attr_eps：
                属性阈值
            density_metric:
                密度计算方式，默认为截断密度，支持 gauss
            dist_metric：
                距离计算方法，默认为 euclidean，支持['euclidean','braycurtis', 'canberra', 'chebyshev', 'cityblock',
                'correlation', 'cosine', 'dice',  'hamming','jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
                'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.]
            algorithm：
                近邻计算方法，计算最近邻点及距离，默认'kd_tree',支持['kd_tree','ball_tree','brute','Annoy','HNSW']
            knn_num:
                近邻个数
            leaf_size:
                近邻计算方法中用到的叶子节点个数，会影响计算和查询速度
            connect_eps:
                密度连通性阈值
        '''
        start=time.clock()
        try:
            data = check_array(self.X,allow_nd=True)
        except:
            raise ValueError("Three-dimensional matrix is needed, latitude, longitude and time.")
        #* 立方体数据转二维矩阵，行为样本，列为要素
        data_not_none,pos_not_none = myutil.rasterCube_to_sampleArray(data)
        self.data_not_none = data_not_none
        mixin_near_matrix = myutil.calc_homo_near_cube(data,spatial_eps,time_eps,attr_eps,pos_not_none)
        density = myutil.calc_gaus_density_st(data,spatial_eps,time_eps,attr_eps)
        denser_pos,denser_dist,density_and_k_relation = myutil.calc_repulsive_force(data_not_none,density,knn_num,leaf_size,fast=True)
        if(-1 in denser_pos):
            raise ValueError('阈值太小啦~,或者尝试使用高斯密度呢：density_metric=gauss')
            pass
        gamma = myutil.calc_gamma(density,denser_dist)
        labels, core_points = myutil.extract_cluster_auto_st(data_not_none,density,connect_eps,denser_dist,denser_pos,gamma,mixin_near_matrix)
        self.labels = labels
        self.core_points = core_points
        self.density_and_k_relation = density_and_k_relation
        self.density = density
        end=time.clock()
        self.calc_time = str(end-start)
        return self


#%%
#* 测试

import xarray as xr
import os
import numpy as np
filePath = os.path.join(r'Z:\regions_daily_010deg\\05\\2015.nc')
dataset = xr.open_dataset(filePath)
pre_ds = dataset['precipitation']
lon = pre_ds.lon
lat = pre_ds.lat
# time = pre_ds.time
lon_range = lon[(lon>-30)&(lon<70)]
lat_range = lat[(lat>30)&(lat<90)]
var = pre_ds.sel(lon=lon_range,lat = lat_range)
var = var.resample(time='1M',skipna=True).sum()
var_t = var
reduced = var_t.coarsen(lon=5).mean().coarsen(lat=5).mean()
data_nc = np.array(reduced)
# times,rows,cols = data_nc.shape
# data_not_none = np.zeros((times*rows*cols,4))
# data_all = np.zeros((times*rows*cols,4))
# num = 0
# for i in range(rows):
#     for j in range(cols):
#         for k in range(times):
#             data_all[num,:] = [i,j,k,data_nc[k,i,j]]
#             num+=1
#         pass
#     pass
# # data[:,[0,1]]=data[:,[1,0]]
# not_none_pos = np.where(data_all[:,3]!=0)[0] #* 去除零值后的数据，在全局的位置 [638,629,1004,……] 值为 data_all数据下标
# nan_pos = np.where(np.isnan(data_all[:,3]))[0] #* 获取 值为 nan 的下标
# not_none_pos = np.setdiff1d(not_none_pos,nan_pos)
# data_not_none = data_all[not_none_pos]
# pos_not_none = np.full((times*rows*cols),-1,dtype=np.int64) #* 全局数据中，不为零的下标[-1,-1,0,-1,1,-1,2,3,4,……] 值为 data_not_none 下标
# pos_not_none[not_none_pos] = np.array(range(len(not_none_pos)))

#%%
from importlib import reload
reload(myutil)

spatial_eps=6
time_eps = 1
attr_eps=30
density_metric='gauss'
leaf_size=3000
knn_num=100
spre = ADPTC(data_nc)
spre.st_clustering_raster(spatial_eps,time_eps,attr_eps,density_metric,knn_num=knn_num,leaf_size=leaf_size,connect_eps=0.9)

# visual.show_result(spre.labels,data,np.array(list(spre.core_points)))
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
# cur_label_cell_pos = np.where(spre.labels==1)
fig,axes = plt.subplots()
for i in np.unique(spre.labels):
    if(i==-1):
        continue
    cur_label_cell_pos = np.where(spre.labels==i)
    print("当前类别：",i)
    print("个数：",len(spre.data_not_none[cur_label_cell_pos]))
    print("最大值：",spre.data_not_none[cur_label_cell_pos,3].max())
    print("最小值：",spre.data_not_none[cur_label_cell_pos,3].min())
    print("标准差：",spre.data_not_none[cur_label_cell_pos,3].std())
    print("均值：",spre.data_not_none[cur_label_cell_pos,3].mean())
    print("中位数：",np.median(spre.data_not_none[cur_label_cell_pos,3]))
    print("**************")
    axes.boxplot(x=spre.data_not_none[cur_label_cell_pos,3][0],sym='rd',positions=[i],showfliers=False)
# %%
#** 数据写入 excel
import pandas as pd

cluster_res = np.c_[spre.data_not_none,spre.labels]
data_df = pd.DataFrame(cluster_res)
data_df.columns = ['lon','lat','pre','label']
data_df[['lon','lat','pre']].astype('float')
data_df[['label']].astype('float').astype(int)
data_df.to_excel('../../data/cluster_res_2016_1month.xlsx')

#%%
#* 聚类结果转换为netcdf
def labeled_res_to_netcdf(ori_nc,ori_ndarray,data_table,labels):
    #* 将聚类结果写入DataArray
    dr_labels = np.full(ori_ndarray.shape,-2)
    for i in range(len(data_table)):
        dr_labels[int(data_table[i][1])][int(data_table[i][0])] = labels[i]
        pass
    labeled_res= xr.DataArray(
        dr_labels,
        coords=ori_nc.coords,
        dims=ori_nc.dims
    )
    ds = xr.Dataset(data_vars = dict(label=labeled_res,attr=ori_nc))
    return ds
res = labeled_res_to_netcdf(reduced,data_nc,spre.data_not_none,spre.labels)
#%%
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
def create_map_label(ax):
    # * 创建画图空间
    #* 设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1.2,color='k',alpha=0.5,linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(-30,70,15))
    # gl.xlocator = mticker.FixedLocator([-30,-15,0,15,30,45,60])
    gl.xlabel_style = {'fontsize': 30}
    gl.ylabel_style = {'fontsize': 30}
    return ax 

#* 聚类结果可视化
def showlabel(da,labels):
    proj = ccrs.TransverseMercator(central_longitude=20.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10), subplot_kw={'projection':proj})
    cbar_kwargs={
        'label': 'Lables',
        'ticks':np.arange(-1,np.max(labels),1), 
    }
    levels = np.arange(-1,np.max(labels),1)
    
    ax = create_map_label(ax)
    ncolors = 256
    color_array = plt.get_cmap('Spectral_r')(range(ncolors))
    color_array[0] = [0,0,0,0]
    map_object = LinearSegmentedColormap.from_list(name='my_spectral_r',colors=color_array)
    pre_pic = da.plot.contourf(ax=ax,levels=levels, cmap=map_object, extend='both', cbar_kwargs = cbar_kwargs,transform=ccrs.PlateCarree())
    ax.set_title(' ', fontsize=30)
    # ax.set_aspect('equal') 
    cb = pre_pic.colorbar
    cb.ax.set_ylabel('类簇',fontsize=30)
    cb.ax.tick_params(labelsize=24)
    ax.coastlines()
    fig.show()
 #%%
showlabel(res['label'],spre.labels)
# %%
from importlib import reload
reload(myutil)
reload(visual)
visual.show_result_2d(reduced,spre.labels)
# %%
data_nc.shape
# %%
reload(visual)
reload(myutil)
res = myutil.labeled_res_to_netcdf(reduced,data_nc,spre.data_not_none,spre.labels)
visual.show_result_3d(res['label'],[-30, 70, 30, 90],10)
# %%
reload(visual)
visual.show_box(spre.data_not_none,spre.labels,3,[11,14,15,18,19,21])
# %%
X = spre.data_not_none[np.where(spre.data_not_none[:,2]==5)[0]]
tend = prepare.calc_hopkins(X,1000)
# %%
tend
# %%
prepare.draw_ivat(spre.data_not_none[10000:11000])
# %%
