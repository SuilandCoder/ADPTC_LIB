'''
Description: 可视化结果
Author: SongJ
Date: 2020-12-29 14:47:41
LastEditTime: 2021-04-12 10:44:29
LastEditors: SongJ
'''
import matplotlib.pyplot as plt
import numpy as np
from . import myutil
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D


#*************************************** 结果展示*******************************************#
# 绘制密度和时间距离图
def showDenDisAndDataSet(den, dis):
    # 密度和时间距离图显示在面板
    plt.figure(num=1, figsize=(15, 9))
    ax1 = plt.subplot(121)
    plt.scatter(x=den, y=dis, c='k', marker='o', s=30)
    for i in range(len(den)):
        plt.text(den[i],dis[i],i,fontdict={'fontsize':16})
    plt.xlabel('Density')
    plt.ylabel('Distance')
    plt.title('Decision Diagram')
    plt.sca(ax1)
    plt.show()


# 确定类别点,计算每点的密度值与最小距离值的乘积，并画出决策图，以供选择将数据共分为几个类别
def show_nodes_for_chosing_mainly_leaders(gamma):
    plt.figure(num=2, figsize=(15, 10))
    # -np.sort(-gamma) 将gamma从大到小排序
    y=-np.sort(-gamma)
    indx = np.argsort(-gamma)
    plt.scatter(x=range(len(gamma)), y=y, c='k', marker='o', s=50)
    for i in range(int(len(y))):
        plt.text(i,y[i],indx[i],fontdict={'fontsize':16},c='#f00')
    plt.xlabel('n',fontsize=20)
    plt.ylabel('γ',fontsize=20)
    # plt.title('递减顺序排列的γ')
    plt.show()


def show_result(labels, data, corePoints=[]):
    # 画最终聚类效果图
    plt.figure(num=3, figsize=(15, 10))
    # 一共有多少类别
    clusterNum = np.unique(labels)
    scatterColors = [
            '#FF0000','#FFA500','#00FF00','#228B22',
            '#0000FF','#FF1493','#EE82EE','#000000',
            '#00FFFF','#F099C0','#0270f0','#96a9f0',
            '#99a9a0','#22a9a0','#a99ff9','#a90ff9'
    ]

    # 绘制分类数据
    for i in clusterNum:
        if(i==-1 or i==-2):
            colorSytle = '#510101'
            subCluster = data[np.where(labels == i)]
            plt.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=80, marker='*', alpha=1)
            continue
        # 为i类别选择颜色
        colorSytle = scatterColors[i % len(scatterColors)]
        # 选择该类别的所有Node
        subCluster = data[np.where(labels == i)]
        plt.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=25, marker='o', alpha=1,label=i)
    # 绘制每一个类别的聚类中心
    if(len(corePoints)!=0):
        plt.scatter(x=data[corePoints, 0], y=data[corePoints, 1], marker='+', s=300, c='k', alpha=1)
    # plt.title('聚类结果图')
    plt.legend(loc='upper left',fontsize='18')
    plt.tick_params(labelsize=18)
    plt.show()

def show_data(data):
    plt.figure(num=3, figsize=(15, 10))
    plt.scatter(data[:,0],data[:,1],s=300,edgecolor='')
    for i in range(len(data)):
        plt.text(data[i,0],data[i,1],i,fontdict={'fontsize':16})
    plt.show()



# 绘制密度和时间距离图
def showDenDisAndDataSet_label(den, dis,labels,font_show = True):
        # 一共有多少类别
    clusterNum = np.unique(labels)
    scatterColors = [
            '#FF0000', '#FFA500', '#228B22',
            '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
                '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]
    # 密度和时间距离图显示在面板
    plt.figure(num=1, figsize=(15, 9))
    ax1 = plt.subplot(121)
    # 绘制分类数据
    for i in clusterNum:
        if(i==-1 or i==-2):
            colorSytle = '#510101'
            subCluster_id = np.where(labels == i)[0]
            plt.scatter(den[subCluster_id], dis[subCluster_id], c=colorSytle, s=200, marker='*', alpha=1)
            continue
        # 为i类别选择颜色
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster_id = np.where(labels == i)[0]
        plt.scatter(x=den[subCluster_id], y=dis[subCluster_id], c=colorSytle, s=200, marker='o', alpha=1)
    # 绘制每一个类别的聚类中心
    # plt.scatter(x=den[corePoints], y=dis[corePoints], marker='+', s=200, c='k', alpha=1)
    if font_show == True:
        for i in range(len(den)):
            plt.text(den[i],dis[i],i,fontdict={'fontsize':16})
    plt.xlabel('Density - ρ',fontdict={'fontsize':22})
    plt.ylabel('Distance - δ',fontdict={'fontsize':22})
    plt.title('Decision Diagram',fontdict={'fontsize':22})
    plt.sca(ax1)
    plt.show()


# 确定类别点,计算每点的密度值与最小距离值的乘积，并画出决策图，以供选择将数据共分为几个类别
def show_nodes_for_chosing_mainly_leaders_label(gamma,labels,font_show = True):
    # 一共有多少类别
    clusterNum = np.unique(labels)
    scatterColors = [
            '#FF0000', '#FFA500', '#228B22',
            '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
                '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]
    plt.figure(num=2, figsize=(15, 10))
    # -np.sort(-gamma) 将gamma从大到小排序
    y=-np.sort(-gamma)
    indx = np.argsort(-gamma)
    # 绘制分类数据
    for i in clusterNum:
        if(i==-1 or i==-2):
            colorSytle = '#510101'
            subCluster_id = np.where(labels == i)[0]
            ori_indx = [np.where(indx==i)[0][0] for i in subCluster_id]
            plt.scatter(x=ori_indx, y=gamma[subCluster_id], c=colorSytle, s=200, marker='*', alpha=1)
            continue
        # 为i类别选择颜色
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster_id = np.where(labels == i)[0]
        ori_indx = [np.where(indx==i)[0][0] for i in subCluster_id]
        plt.scatter(x=ori_indx, y=gamma[subCluster_id], c=colorSytle, s=200, marker='o', alpha=1)
    # plt.scatter(x=range(len(gamma)), y=y, c='k', marker='o', s=50)
    if font_show == True:
        for i in range(int(len(y))):
            plt.text(i,y[i],indx[i],fontdict={'fontsize':16},c='#000')
    plt.xlabel('n',fontsize=20)
    plt.ylabel('γ',fontsize=20)
    # plt.title('递减顺序排列的γ')
    plt.show()

def show_data_label(data,labels,font_show = True):
        # 一共有多少类别
    clusterNum = np.unique(labels)
    scatterColors = [
            '#FF0000', '#FFA500', '#228B22',
            '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
                '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]
    plt.figure(num=3, figsize=(15, 10))
        # 绘制分类数据
    for i in clusterNum:
        if(i==-1 or i==-2):
            colorSytle = '#510101'
            subCluster_id = np.where(labels == i)[0] 
            plt.scatter(x=data[subCluster_id,0], y=data[subCluster_id,1], c=colorSytle, s=200, marker='*', alpha=1)
            continue
        # 为i类别选择颜色
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster_id = np.where(labels == i)[0] 
        plt.scatter(x=data[subCluster_id,0], y=data[subCluster_id,1], c=colorSytle, s=200, marker='o', alpha=1)
    
    # plt.scatter(data[:,0],data[:,1],s=300,edgecolor='')
    if font_show == True:
        for i in range(len(data)):
            plt.text(data[i,0],data[i,1],i,fontdict={'fontsize':16})
    plt.show()
    
    

def create_map_label(ax):
    # * 创建画图空间
    #* 设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1.2,color='k',alpha=0.5,linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # gl.xlocator = mticker.FixedLocator(np.arange(-30,70,15))
    # gl.xlocator = mticker.FixedLocator([-30,-15,0,15,30,45,60])
    gl.xlabel_style = {'fontsize': 30}
    gl.ylabel_style = {'fontsize': 30}
    return ax 

#* 聚类结果可视化
def showlabel(da,labels):
    # proj = ccrs.TransverseMercator(central_longitude=20.0)
    proj = ccrs.PlateCarree()
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
    cb.ax.set_ylabel('labels',fontsize=30)
    cb.ax.tick_params(labelsize=24)
    ax.coastlines()
    fig.show()
    pass
    
def show_result_2d(ori_nc,labels):
    myutil.check_netcdf(ori_nc)
    data_nc = np.array(ori_nc)
    data_not_none,pos_not_none = myutil.rasterArray_to_sampleArray(data_nc)
    label_nc = myutil.labeled_res_to_netcdf(ori_nc,data_nc,data_not_none,labels)
    showlabel(label_nc['label'],labels)
    pass

def show_result_3d(ori_nc,adptc,extent,time_extent,label,path=''):
    '''
        ori_nc: 包含类别标签的 netcdf 数据（xarray.DataArray）
        adptc：聚类结果对象
        extent: 经纬度范围  [lon,lon,lat,lat]
        time_extent:时间范围 [time1,time2]
        label：要显示的类别
    '''
    res = myutil.labeled_res_to_netcdf(ori_nc,adptc.data_not_none,adptc.labels)
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


#*展示箱图
def show_box(data,labels,attr_index,labels_show):
    '''
        data: 样本点数据集合 data_not_none
        labels: 聚类结果
        attr_index: 非时空属性下标
        labels_show:要展示的类别标签集合
    '''
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots()
    for i in range(len(labels_show)):
        cur_label_cell_pos = np.where(labels==labels_show[i])
        axes.boxplot(x=data[cur_label_cell_pos,attr_index][0],sym='rd',positions=[i],showfliers=False,notch=True)
    plt.xlabel('label',fontsize=20)  # x轴标注
    plt.ylabel('attr',fontsize=20)  # y轴标注
    plt.xticks(range(len(labels_show)),labels_show)
    plt.tick_params(labelsize=15)
    pass


#* 展示小提琴图
def show_vlines(data,labels,attr_index,labels_show):
    '''
        data: 样本点数据集合 data_not_none
        labels: 聚类结果
        attr_index: 非时空属性下标
        labels_show:要展示的类别标签集合
    '''
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots()
    for i in range(len(labels_show)):
        cur_label_cell_pos = np.where(labels==labels_show[i])
        axes.violinplot(data[cur_label_cell_pos,attr_index][0],positions=[i],showmeans=False,showmedians=True)
    plt.xlabel('label',fontsize=20)  # x轴标注
    # plt.xticks(labels_show)
    plt.ylabel('attr',fontsize=20)  # x轴标注
    plt.xticks(range(len(labels_show)),labels_show)
    plt.tick_params(labelsize=15)
    pass