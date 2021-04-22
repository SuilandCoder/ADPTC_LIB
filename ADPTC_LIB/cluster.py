'''
Description:
Author: SongJ
Date: 2020-12-28 10:23:45
LastEditTime: 2021-04-12 10:33:11
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
