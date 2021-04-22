'''
Description:  测试 clustering 功能
Author: SongJ
Date: 2020-12-28 17:11:24
LastEditTime: 2021-04-12 10:00:41
LastEditors: SongJ
'''
import sys

#%%
import numpy as np

sys.path.insert(0,r"D:\workspace_clustering\ATDPC_Package\ADPTC_LIB")
import visual
from cluster import ADPTC

X = np.loadtxt(r"../test_data/Aggregation.txt", delimiter="\t")
X = X[:,[0,1]]

# X = np.loadtxt(r"../test_data/gaussian_point.txt", delimiter=" ")
print(X)
# %%
atdpc_obj = ADPTC(X)
atdpc_obj.clustering(2,density_metric='gauss',knn_num=20,connect_eps=0.9,leaf_size=30,fast=False)
print(atdpc_obj.labels)
visual.show_result(atdpc_obj.labels,X,np.array(list(atdpc_obj.core_points)))
import sys

#%%
import numpy as np

sys.path.insert(0,r"D:\workspace_clustering\ATDPC_Package\ATDPC")
from ADPTC_LIB import visual
from ADPTC_LIB.cluster import ADPTC

X = np.loadtxt(r"../test_data/Aggregation.txt", delimiter="\t")
X = X[:,[0,1]]
atdpc_obj = ADPTC(X)
atdpc_obj.clustering(2)
visual.show_result(atdpc_obj.labels,X,np.array(list(atdpc_obj.core_points)))
# %%
import ADPTC_LIB
# from ADPTC_LIB.cluster import ADPTC
# %%
from ADPTC_LIB.cluster import ADPTC
from ADPTC_LIB import visual
# %%
import numpy as np
X = np.loadtxt(r"../test_data/Aggregation.txt", delimiter="\t")
X = X[:,[0,1]]
atdpc_obj = ADPTC(X)
atdpc_obj.clustering(2)
visual.show_result(atdpc_obj.labels,X,np.array(list(atdpc_obj.core_points)))
# %%
import xarray as xr
import os
import numpy as np
filePath = os.path.join(r'Z:\regions_daily_010deg\\05\\2013.nc')
dataset = xr.open_dataset(filePath)
pre_ds = dataset['precipitation']
lon = pre_ds.lon
lat = pre_ds.lat
lon_range = lon[(lon>-30)&(lon<70)]
lat_range = lat[(lat>30)&(lat<90)]
var = pre_ds.sel(lon=lon_range,lat = lat_range)
var = var.resample(time='1M',skipna=True).sum()
var_t = var.sel(time=var.time[0])
reduced = var_t.coarsen(lon=5).mean().coarsen(lat=5).mean()
data_nc = np.array(reduced)
# %%
spatial_eps=4
attr_eps=8
density_metric='gauss'
spre = ADPTC(data_nc)
spre.spacial_clustering_raster(spatial_eps,attr_eps,density_metric,knn_num=100,leaf_size=3000,connect_eps=0.9)
visual.show_result_2d(reduced,spre.labels)
# %%
import xarray as xr
import numpy as np
temp= xr.open_dataset(r'Z:\MSWX\temp\2020.nc')
temp_2020 = temp['air_temperature']
lon = temp_2020.lon
lat = temp_2020.lat
time = temp_2020.time
lon_range = lon[(lon>70)&(lon<140)]
lat_range = lat[(lat>15)&(lat<55)]
var = temp_2020.sel(lon=lon_range,lat = lat_range)
reduced = var.coarsen(lon=5).mean().coarsen(lat=5).mean()
data_nc = np.array(reduced)
# %%
s_eps = 5
t_eps = 1
attr_eps = 2.5
spre = ADPTC(data_nc)
spre.st_clustering_raster(s_eps,t_eps,attr_eps,density_metric,knn_num=100,leaf_size=3000,connect_eps=0.9)

# %%
# from ADPTC_LIB import myutil
# res = myutil.labeled_res_to_netcdf(reduced,data_nc,spre.data_not_none,spre.labels)
from . import myutil
from importlib import reload
reload(myutil)
show_result_3d(reduced,spre,[70, 140, 15, 50],[0,12],21)

# %%

#* 聚类结果转换为netcdf
def labeled_res_to_netcdf(ori_nc,data_table,labels):
    #* 将聚类结果写入DataArray
    ori_ndarray = np.array(ori_nc)
    dr_labels = np.full(ori_ndarray.shape,-2)
    for i in range(len(data_table)):
        if(ori_ndarray.ndim==2):
            dr_labels[int(data_table[i][1])][int(data_table[i][0])] = labels[i]
        elif(ori_ndarray.ndim==3):
            dr_labels[int(data_table[i][2])][int(data_table[i][0])][int(data_table[i][1])] = labels[i]
        else:
            raise ValueError("Two or Three-dimensional matrix is needed")
        pass
    labeled_res= xr.DataArray(
        dr_labels,
        coords=ori_nc.coords,
        dims=ori_nc.dims
    )
    ds = xr.Dataset(data_vars = dict(label=labeled_res,attr=ori_nc))
    return ds


def show_result_3d(ori_nc,adptc,extent,time_extent,label,path=''):
    '''
        ori_nc: 包含类别标签的 netcdf 数据（xarray.DataArray）
        adptc：聚类结果对象
        extent: 经纬度范围  [lon,lon,lat,lat]
        time_extent:时间范围 [time1,time2]
        label：要显示的类别
    '''
    res = labeled_res_to_netcdf(ori_nc,adptc.data_not_none,adptc.labels)
    label_nc = res['label']
    data_nc = np.array(label_nc)
    times,rows,cols = data_nc.shape
    data_not_none = np.zeros((times*rows*cols,4))
    data_all = np.zeros((times*rows*cols,4))
    num = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(times):
                data_all[num,:] = [i,j,k,data_nc[k,i,j]]
                num+=1
            pass
        pass
    data = data_all[np.where(data_all[:,3]==label)[0]]
    lons = []
    lats = []
    times = []
    for i in range(len(data)):
        lons.append(float(label_nc.lon[int(data[i,1])].values))
        lats.append(float(label_nc.lat[int(data[i,0])].values))
        times.append(int(data[i,2]))
        pass
    colors = ["#CC5C5C","#CF0000","#D80000","#E90000","#FC0000","#FF5100","#FF9D00","#FFBD00","#FBD500","#F0EA00","#D8F500","#B0FF00","#49FF00","#00F700","#00E400","#00CF00","#00BA00","#00A700","#009D1D","#00A668","#00AA95","#00AAA8","#00A0C7","#0095DD","#0080DD","#0054DD","#0009DD","#0000C5","#0D00A8","#58009F","#830094","#7A008B","#41004B"]
    draw_geo3dscatter(lons,lats,times,colors[i%len(colors)],extent,time_extent,path)
    pass

def draw_geo3dscatter(lons,lats,times,label_color,extent,time_extent,path=''):
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    # Create a basemap instance that draws the Earth layer
    bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
                urcrnrlon=extent[1], urcrnrlat=extent[3],
                projection='cyl', resolution='l', fix_aspect=False, ax=ax)
    ax.add_collection3d(bm.drawcoastlines(linewidth=0.25))
    ax.view_init(azim=250, elev=20)
    ax.set_xlabel('Longitude (°E)', labelpad=20,fontsize=30)
    ax.set_ylabel('Latitude (°N)', labelpad=20,fontsize=30)
    ax.set_zlabel('Month', labelpad=10,fontsize=30)
    # Add meridian and parallel gridlines
    lon_step = 10
    lat_step = 15
    meridians = np.arange(extent[0], extent[1] + lon_step, lon_step)
    parallels = np.arange(extent[2], extent[3] + lat_step, lat_step)
    ax.set_yticks(parallels)
    ax.set_yticklabels(parallels)
    ax.set_xticks(meridians)
    ax.set_xticklabels(meridians)
    ax.set_zlim(time_extent[0], time_extent[1])
    # ax1 = fig.add_subplot(projection='3d')
    ax.scatter(lons, lats, times,c=label_color,alpha=0.5,s=20,marker='s')
    ax.scatter(lons, lats,c='#00000005',alpha=0.5,s=20,marker='o')
    plt.tick_params(labelsize=25)
    if(path!=''):
        plt.savefig(path)
    plt.show()
    pass
