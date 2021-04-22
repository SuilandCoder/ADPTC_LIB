'''
Description: 工具
Author: SongJ
Date: 2020-12-28 14:10:28
LastEditTime: 2021-04-10 11:18:43
LastEditors: SongJ
'''
import time
import xarray as xr
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import jit, njit
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import BallTree, DistanceMetric, KDTree
from . import DPTree
# from DPTree import DPTree, label_these_node, split_cluster
# from DPTree_ST import (DPTree, label_these_node, label_these_node_new,
#                        label_these_node_new2, split_cluster_new, split_outlier)
from . import DPTree_ST

def fn_timer(*args,**kwargs):
    def mid_fn(function):
        def function_timer(*in_args, **kwargs):
            t0 = time.time()
            result = function(*in_args, **kwargs)
            t1 = time.time()
            print (" %s: %s seconds" %
                (args[0], str(t1-t0))
                )
            return result
        return function_timer
    return mid_fn

def check_netcdf(X):
    if type(X)!=xr.DataArray:
        raise ValueError("Only support datatype DataArray of xarray, please handle netcdf data by the library xarray.")

# 二维栅格数据转换为样本点数据，每个网格为一个样本点
def rasterArray_to_sampleArray(data):
    rows,cols = data.shape
    data_all = np.zeros((rows*cols,3))
    num = 0
    for i in range(rows):
        for j in range(cols):
            data_all[num,:] = [i,j,data[i,j]]
            num+=1
            pass
        pass
    data_all[:,[0,1]]=data_all[:,[1,0]]
    not_none_pos = np.where(data_all[:,2]!=0)[0] #* 去除零值后的数据，在全局的位置 [638,629,1004,……] 值为 data_all数据下标
    nan_pos = np.where(np.isnan(data_all[:,2]))[0] #* 获取 值为 nan 的下标
    not_none_pos = np.setdiff1d(not_none_pos,nan_pos)
    data_not_none = data_all[not_none_pos]
    pos_not_none = np.full((rows*cols),-1,dtype=np.int64) #* 全局数据中，不为零的下标[-1,-1,0,-1,1,-1,2,3,4,……] 值为 data_not_none 下标
    pos_not_none[not_none_pos] = np.array(range(len(not_none_pos)))
    return data_not_none,pos_not_none


# 三维时空立方体转换为样本点矩阵，每个单元立方格为一个样本点
def rasterCube_to_sampleArray(data):
    times,rows,cols = data.shape
    data_not_none = np.zeros((times*rows*cols,4))
    data_all = np.zeros((times*rows*cols,4))
    num = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(times):
                data_all[num,:] = [i,j,k,data[k,i,j]]
                num+=1
            pass
        pass
    # data[:,[0,1]]=data[:,[1,0]]
    not_none_pos = np.where(data_all[:,3]!=0)[0] #* 去除零值后的数据，在全局的位置 [638,629,1004,……] 值为 data_all数据下标
    nan_pos = np.where(np.isnan(data_all[:,3]))[0] #* 获取 值为 nan 的下标
    not_none_pos = np.setdiff1d(not_none_pos,nan_pos)
    data_not_none = data_all[not_none_pos]
    pos_not_none = np.full((times*rows*cols),-1,dtype=np.int64) #* 全局数据中，不为零的下标[-1,-1,0,-1,1,-1,2,3,4,……] 值为 data_not_none 下标
    pos_not_none[not_none_pos] = np.array(range(len(not_none_pos)))
    return data_not_none,pos_not_none


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



@fn_timer("计算距离矩阵")
def calc_dist_matrix(data, metric='euclidean'):
    dist_mat = squareform(pdist(data, metric=metric))
    return dist_mat

def calc_attr_dist_mat(data,attrs_index,metric='euclidean'):
    rows = data.shape[0]
    try:
        attr_dist_mat = squareform(pdist(data[:,attrs_index].reshape(rows, 1),metric=metric))
    except:
        attr_dist_mat = squareform(pdist(data[:,attrs_index].reshape(rows, 1),metric='euclidean'))
    return attr_dist_mat

@fn_timer("计算截断密度")
@njit
def calcu_cutoff_density(dist_mat, eps):
    '''
    计算截断密度
    '''
    local_cutoff_density = np.where(dist_mat < eps, 1, 0).sum(axis=1)
    local_cutoff_density = local_cutoff_density
    return local_cutoff_density

@fn_timer("计算空间属性邻近域")
def calc_homo_near_grid(data,s_eps,attr_eps,pos_not_none):
    '''
        获取点的空间属性邻近域 mixin_near_matrix
    '''
    mixin_near_matrix = {}
    rows,cols = data.shape
    num = 0
    for i in range(rows):
        for j in range(cols):
            #* 计算每个点的邻域范围:
            left_lon = i-s_eps if i-s_eps>=0 else 0
            rigth_lon = i+s_eps if i+s_eps<rows else rows
            up_lat = j-s_eps if j-s_eps>=0 else 0
            down_lat = j+s_eps if j+s_eps<cols else cols
            s_near = data[left_lon:rigth_lon+1,up_lat:down_lat+1]
            if(data[i,j]!=0  and (not np.isnan(data[i,j]))):
                pos_s_near = np.where((np.abs(s_near-data[i,j])<=attr_eps) & (s_near!=0) &(~np.isnan(s_near)))
                pos_data = np.vstack(pos_s_near) + np.array([[left_lon],[up_lat]])
                pos_in_matrix = cols*pos_data[0]+pos_data[1]  #* 获取全局邻域位置（全局包含空值点）
                pos = pos_not_none[pos_in_matrix]
                mixin_near_matrix[num] = pos
                num+=1
            pass
        pass
    return mixin_near_matrix


def calc_homo_near_cube(data,s_eps,t_eps,attr_eps,pos_not_none):
    '''
        获取点的时空属性邻近域 mixin_near_matrix
    '''
    mixin_near_matrix = {}
    time_len,rows,cols = data.shape
    num = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(time_len):
                #* 计算每个点的邻域范围:
                left_lon = i-s_eps if i-s_eps>=0 else 0
                rigth_lon = i+s_eps if i+s_eps<rows else rows
                up_lat = j-s_eps if j-s_eps>=0 else 0
                down_lat = j+s_eps if j+s_eps<cols else cols
                early_time = k-t_eps if k-t_eps>=0 else 0
                lated_time = k+t_eps if k+t_eps<time_len else time_len
                s_near = data[early_time:lated_time+1,left_lon:rigth_lon+1,up_lat:down_lat+1]
                # s_near = s_near[np.where(~np.isnan(s_near) & (s_near!=0))]
                if(data[k,i,j]!=0  and (not np.isnan(data[k,i,j]))):
                    pos_s_near = np.where((np.abs(s_near-data[k,i,j])<=attr_eps) & (s_near!=0) &(~np.isnan(s_near)))
                    pos_data = np.vstack(pos_s_near) + np.array([[early_time],[left_lon],[up_lat]])
                    pos_in_matrix = time_len*cols*pos_data[1]+time_len*pos_data[2]+pos_data[0]  #* 获取全局邻域位置（全局包含空值点）
                    pos = pos_not_none[pos_in_matrix]
                    mixin_near_matrix[num] = pos
                    num+=1
                pass
            pass
        pass
    return mixin_near_matrix



@fn_timer("计算高斯密度")
def calc_gaus_density_spatial(data,s_eps,attr_eps):
    '''
        此处 data 为空间栅格矩阵数据，行列分别为：lon,lat
    '''
    rows,cols = data.shape
    zero_num = np.where(data==0,1,0).sum()
    nan_num = np.where(np.isnan(data),1,0).sum()
    density_list_len = rows*cols - zero_num - nan_num
    density = np.zeros(density_list_len,dtype=np.float32)
    num = 0
    for i in range(rows):
        for j in range(cols):
            #* 计算每个点的邻域范围:
            left_lon = i-s_eps if i-s_eps>=0 else 0
            rigth_lon = i+s_eps if i+s_eps<rows else rows
            up_lat = j-s_eps if j-s_eps>=0 else 0
            down_lat = j+s_eps if j+s_eps<cols else cols
            s_near = data[left_lon:rigth_lon+1,up_lat:down_lat+1]
            s_near = s_near[np.where((~np.isnan(s_near)) & (s_near!=0))]
            if(data[i,j]!=0 and (not np.isnan(data[i,j]))):
                density[num] = np.exp(-1*((1-(np.abs(s_near-data[i,j])))/attr_eps)**2).sum()
                num+=1
            pass
        pass
    return density


@fn_timer("密度计算")
# @njit
def calc_gaus_density_st(data,s_eps,t_eps,attr_eps):
    '''
        此处 data 为立方体数据，三个维度：time,lon,lat
    '''
    time_len,rows,cols = data.shape
    zero_num = np.where(data==0,1,0).sum()
    nan_num = np.where(np.isnan(data),1,0).sum()
    density_list_len = time_len*rows*cols - zero_num - nan_num
    density = np.zeros(density_list_len,dtype=np.float32)
    num = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(time_len):
                #* 计算每个点的邻域范围:
                left_lon = i-s_eps if i-s_eps>=0 else 0
                rigth_lon = i+s_eps if i+s_eps<rows else rows
                up_lat = j-s_eps if j-s_eps>=0 else 0
                down_lat = j+s_eps if j+s_eps<cols else cols
                early_time = k-t_eps if k-t_eps>=0 else 0
                lated_time = k+t_eps if k+t_eps<time_len else time_len
                s_near = data[early_time:lated_time+1,left_lon:rigth_lon+1,up_lat:down_lat+1]
                s_near = s_near[np.where((~np.isnan(s_near)) & (s_near!=0))]
                if(data[k,i,j]!=0 and (not np.isnan(data[k,i,j]))):
                    density[num] = np.exp(-1*((1-(np.abs(s_near-data[k,i,j])))/attr_eps)**2).sum()
                    num+=1
                pass
            pass
        pass
    return density


@fn_timer("计算空间近邻")
def calc_spatial_neighbor(X_spatial,eps,leaf_size):
    '''
        使用 kdtree 计算空间近邻
        主要是借助kdtree解决大数据量的计算问题
    '''
    tree = KDTree(X_spatial, leaf_size=leaf_size)
    ind = tree.query_radius(X_spatial, eps, return_distance=False, count_only=False, sort_results=False)
    return ind

@fn_timer("计算时空邻居")
# @njit
def calc_st_neighbor(X_time,eps,spatial_neighbor):
    '''
        计算时间近邻
    '''
    st_neighbor = []
    flattened_time = X_time.flatten()
    rows = len(flattened_time)
    for i in range(rows):
        cur_spat_neighbor = spatial_neighbor[i]
        st_neighbor.append(cur_spat_neighbor[np.where(abs(flattened_time[cur_spat_neighbor]-flattened_time[i])<=eps)[0]])
        pass
    return np.array(st_neighbor)


# @fn_timer("计算时空邻居")
# def calc_st_neighbor(spatial_neighbor,time_neighbor):
#     st_neighbor={}
#     rows = spatial_neighbor.shape[0]
#     for i in range(rows):
#         st_neighbor[i]=np.intersect1d(spatial_neighbor[i],time_neighbor[i])
#         pass
#     return st_neighbor


@fn_timer("计算混合邻居")
def calc_mixin_near_matrix(space_dist_mat,spatial_eps,attr_dist_mat,attr_eps):
    rows = space_dist_mat.shape[0]
    mixin_near_matrix = {}
    for i in range(rows):
        space_near = np.where(space_dist_mat[i,:]<=spatial_eps)[0]
        attr_near = np.where(attr_dist_mat[i,:]<=attr_eps)[0]
        mixin_near_matrix[i]=np.intersect1d(space_near,attr_near)
    return mixin_near_matrix



# @njit
def calc_gaus_density_njit(rows,ind,dist,st_neighbors,eps):
    local_gaus_density = np.zeros((rows,),dtype=np.float32)
    for i in range(rows):
        arg_intersect_ind = np.where(np.in1d(ind[i],st_neighbors[i]))
        local_gaus_density[i] = np.exp(-1 *(dist[i][arg_intersect_ind]/eps)**2).sum()
    return local_gaus_density

def calc_mixin_near_matrix(rows,ind,st_neighbors):
    mixin_near_matrix = {}
    for i in range(rows):
        arg_intersect_ind = np.where(np.in1d(ind[i],st_neighbors[i]))
        mixin_near_matrix[i] = ind[i][arg_intersect_ind]
    return mixin_near_matrix


@fn_timer("计算高斯密度")
@njit
def calcu_gaus_density_spatial(near_matrix,dist_mat, eps):
    '''
    计算高斯密度
    '''
    rows = dist_mat.shape[0]
    local_gaus_density = np.zeros((rows,),dtype=np.float32)
    for i in range(rows):
        near_nodes = near_matrix[1][np.where(near_matrix[0]==i)] 
        local_gaus_density[i] = np.exp(-1*((1-dist_mat[i][near_nodes])/eps)**2).sum()
    return local_gaus_density
    

@fn_timer("计算高斯密度")
@njit
def calcu_gaus_density(dist_mat, eps):
    '''
    计算高斯密度
    '''
    rows = dist_mat.shape[0]
    local_gaus_density = np.zeros((rows,),dtype=np.float32)
    for i in range(rows):
        local_gaus_density[i] = np.exp(-1 *((dist_mat[i, :])/(eps))**2).sum()
        pass
    return local_gaus_density


def calc_density(dist_mat,eps,density_metric):
    if(density_metric=='gauss'):
        return calcu_gaus_density(dist_mat,eps)
    else:
        return calcu_cutoff_density(dist_mat,eps)


def calc_repulsive_force(data,density,k_num,leaf_size,dist_mat=[],fast=False):
    if(fast):
        denser_pos,denser_dist,density_and_k_relation = calc_repulsive_force_fast(data,k_num,density,leaf_size)
        pass
    else:
        denser_pos,denser_dist,density_and_k_relation = calc_repulsive_force_classical(data,density,dist_mat)
    return denser_pos,denser_dist,density_and_k_relation

@fn_timer("计算斥群值_快速")
def calc_repulsive_force_fast(data, k_num, density, leaf_size):
    #* b. 求每个点的k近邻
    # tree = BallTree(data,leaf_size=2000,metric=DistanceMetric.get_metric('mahalanobis',V=np.cov(data.T)))
    tree = KDTree(data, leaf_size=leaf_size)
    dist, ind = tree.query(data, k=k_num)

    #* 统计 密度 与 k 值的相关性：
    density_and_k_relation = np.zeros((ind.shape[0],2),dtype=np.float32)

    #* c. 计算 k近邻点 是否能找到斥群值
    denser_dist = np.full(ind.shape[0], -1,dtype=np.float32)
    denser_pos = np.full(ind.shape[0],-1,dtype=np.int32)
    for i in range(ind.shape[0]):
        denser_list = np.where(density[ind[i]]>density[i])[0]
        if(len(denser_list)>0):
            denser_dist[i] = dist[i][denser_list[0]]
            denser_pos[i] = ind[i][denser_list[0]] #* 这个pos为data中的下标，没有属性为空的点
            density_and_k_relation[i][0] = density[i]
            density_and_k_relation[i][1] = denser_list[0]
            pass

    #* d. 增加 k值，寻找斥群值:0.
    not_found_data = list(np.where(denser_pos==-1)[0])
    #* 对密度进行排序，剔除密度最大的点
    max_density_idx = not_found_data[np.argmax(density[not_found_data])]
    density[max_density_idx] = density[max_density_idx]+1
    not_found_data.pop(np.argmax(density[not_found_data])) 
    num = 1
    cur_k = k_num
    while(len(not_found_data)>0):
        cur_data_id = not_found_data.pop()
        cur_k = cur_k+k_num
        if(cur_k>=data.shape[0]):
            break
        cur_dist, cur_ind= tree.query(data[cur_data_id:cur_data_id+1], k=cur_k)
        cur_dist, cur_ind = cur_dist[0], cur_ind[0]
        denser_list = np.where(density[cur_ind]>density[cur_data_id])
        while(len(denser_list[0])==0):
            cur_k = cur_k + k_num
            # print("cur_k:",cur_k)
            if(cur_k>=data.shape[0]):
                break
            cur_dist, cur_ind= tree.query(data[cur_data_id:cur_data_id+1], k=cur_k)
            cur_dist, cur_ind = cur_dist[0], cur_ind[0]
            denser_list = np.where(density[cur_ind]>density[cur_data_id])
            pass
        if(len(denser_list[0])>0):
            # print(num)
            num = num+1
            denser_pos[cur_data_id] = cur_ind[denser_list[0][0]]
            denser_dist[cur_data_id] = cur_dist[denser_list[0][0]]
            density_and_k_relation[cur_data_id][0] = density[cur_data_id]
            density_and_k_relation[cur_data_id][1] = denser_list[0][0]
        else:
            print("没找到:",cur_data_id)
        pass
    denser_dist[max_density_idx] = np.max(denser_dist)+1
    denser_pos[max_density_idx] =max_density_idx
    return denser_pos,denser_dist,density_and_k_relation


@fn_timer("计算斥群值_经典")
def calc_repulsive_force_classical(data,density,dist_mat):
    rows = len(data)
    #* 密度从大到小排序
    sorted_density = np.argsort(density)
    #* 初始化，比自己密度大的且最近的距离
    denser_dist = np.zeros((rows,))
    #* 初始化，比自己密度大的且最近的距离对应的节点id
    denser_pos = np.zeros((rows,), dtype=np.int32)
    for index,nodeId in enumerate(sorted_density):
        nodeIdArr_denser = sorted_density[index+1:]
        if nodeIdArr_denser.size != 0:
            #* 计算比当前密度大的点之间距离：
            over_density_sim = dist_mat[nodeId][nodeIdArr_denser]
            #* 获取比自身密度大，且距离最小的节点
            denser_dist[nodeId] = np.min(over_density_sim)
            min_distance_index = np.argwhere(over_density_sim == denser_dist[nodeId])[0][0]
            # 获得整个数据中的索引值
            denser_pos[nodeId] = nodeIdArr_denser[min_distance_index]
        else:
            #* 如果是密度最大的点，距离设置为最大，且其对应的ID设置为本身
            denser_dist[nodeId] = np.max(denser_dist)+1
            denser_pos[nodeId] = nodeId
    return denser_pos,denser_dist,[]


def calc_gamma(density,denser_dist):
    normal_den = density / np.max(density)
    normal_dis = denser_dist / np.max(denser_dist)
    gamma = normal_den * normal_dis
    return gamma


@fn_timer("自动聚类")
def extract_cluster_auto_st(data,density,dc_eps,denser_dist,denser_pos,gamma,mixin_near_matrix):
    '''
        使用 DPTree 进行数据聚类
        dc_eps：density-connectivity 阈值
    '''
    sorted_gamma_index = np.argsort(-gamma)
    tree = DPTree_ST.DPTree()
    tree.createTree(data,sorted_gamma_index,denser_pos,denser_dist,density,gamma)
    outlier_forest, cluster_forest, uncertain_forest=DPTree_ST.split_cluster_new(tree,density,dc_eps,denser_pos,mixin_near_matrix)
    labels,core_points = DPTree_ST.label_these_node_new(outlier_forest,cluster_forest,len(data),uncertain_forest,mixin_near_matrix)
    core_points = np.array(list(core_points))
    labels = labels
    return labels, core_points

    
@fn_timer("自动聚类")
def extract_cluster_auto(data,density,eps,dc_eps,denser_dist,denser_pos,gamma,dist_mat):
    '''
        使用 DPTree 进行数据聚类
        dc_eps：density-connectivity 阈值
    '''
    sorted_gamma_index = np.argsort(-gamma)
    tree = DPTree.DPTree()
    tree.createTree(data,sorted_gamma_index,denser_pos,denser_dist,density,gamma)
    outlier_forest, cluster_forest=DPTree.split_cluster(tree,density,dist_mat,eps,dc_eps,denser_dist)
    labels,core_points = DPTree.label_these_node(outlier_forest,cluster_forest,len(data))
    core_points = np.array(list(core_points))
    labels = labels
    return labels, core_points



