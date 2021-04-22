#%%
import numpy as np
import copy
import matplotlib.pyplot as plt
import time 

def split_cluster_new(tree,local_density,dc_eps,closest_denser_nodes_id,mixin_near_matrix):
    '''
        dc_eps: density_connectivity 阈值
        使用父子节点的直接距离，与子节点与兄弟节点的连通距离进行聚簇划分；
        使用平均密度划分outlier
        返回：
            outlier_forest
            cluster_forest
    '''
    mean_density = np.mean(local_density)
    outlier_forest = {}
    cluster_forest = {}
    uncertain_forest = {}
    not_direct_reach = []
    #* 计算不可直接可达的点：
    for k in range(len(closest_denser_nodes_id)):
        near_nodes = mixin_near_matrix[k]
        if closest_denser_nodes_id[k] not in near_nodes:
            not_direct_reach.append(k)
        pass
    not_direct_reach = np.array(not_direct_reach)
    # not_direct_reach = np.where(closest_dis_denser>eps)[0]
    #* 将不直接距离可达的点按层次排列：
    # not_direct_reach = np.array(not_direct_reach)
    depth_list_not_direct_reach= np.zeros(len(not_direct_reach),dtype=np.int16)
    for i in range(len(not_direct_reach)):
        # depth_list_not_direct_reach[i] = tree.node_dir[not_direct_reach[i]].getLvl()
        depth_list_not_direct_reach[i] = tree.calcu_depth(not_direct_reach[i],0)
        pass
    not_direct_reach = list(not_direct_reach[np.argsort(depth_list_not_direct_reach)])
    #* 模拟栈结构，层次深的先处理
    start = time.clock()
    while(len(not_direct_reach)>0):
        #* 判断是否 连通：距离小于阈值，并且密度要大于子树的平均密度
        node_id = not_direct_reach.pop()
        if(node_id==129193 or node_id==61589 or node_id == 123593):
            print(node_id)
        if node_id in tree.sorted_gamma_index[0:10]:
            cluster_forest[node_id] = tree.remove_subtree(node_id)
            continue
        node = tree.node_dir[node_id]
        parent_id = node.parent_id
        parent_node = tree.node_dir[parent_id]
        children = parent_node.getChildren()
        siblings_reliable = [ i for i in children if i not in not_direct_reach] #* 求得兄弟节点，其中兄弟节点不能是不直接可达的点
        not_reliable_nodes = [i for i in children if i not in siblings_reliable]
        if node_id in not_reliable_nodes:
            not_reliable_nodes.remove(node_id)
        if node_id in siblings_reliable:
            siblings_reliable.remove(node_id)
        pairs_nodes = is_connected_new(tree,local_density,dc_eps,node_id,siblings_reliable,not_reliable_nodes,mixin_near_matrix)
        if len(pairs_nodes)==0:
            if(node_id==tree.root_node.node_id):
                continue
            if(local_density[node_id]-mean_density*dc_eps)>=0:
                #* 获取子节点个数:
                offspring_id = tree.get_subtree_offspring_id(node_id,[node_id])
                if(len(offspring_id)<local_density[node_id]):
                    uncertain_forest[node_id] = tree.remove_subtree(node_id)
                    pass
                else:
                    cluster_forest[node_id] = tree.remove_subtree(node_id)
                    pass
                pass
            else:
                outlier_forest[node_id] = tree.remove_subtree(node_id)
                pass
            pass
        pass
    end = time.clock()
    print('切割树耗时 %s' % str(end - start))
    cluster_forest[tree.root_node.node_id] = tree #* 添加根节点的树
    return outlier_forest, cluster_forest, uncertain_forest


def is_connected_new(tree,local_density,dc_eps,cur_node_id,reliable_nodes,not_reliable_nodes,mixin_near_matrix):
    '''
        cur_node: 当前待判断与父节点连通度的点；
        reliable_nodes：兄弟节点中与父节点直接相连的点；
        not_reliable_nodes：兄弟节点中不与父节点直接相连的点，但可能间接相连；
        连通度判断方案：
            1. 判断 cur_node 与 reliable_nodes 是否可达，是则返回；没有则执行2；
            2. 判断 cur_node 与 not_reliable_nodes(假设为[a,b,c,d,e]) 是否可达，若与[a,b,c]可达，与[d,e]不可达，执行3；
            3. 循环遍历[a,b,c],递归调用本方法 is_connected_entropy(……,cur_node_id=[a],reliable_nodes,not_reliable_nodes=[b,c,d,e])
    '''
    #* 1. 
    if(len(reliable_nodes)==0):
        return []
    for reliable_node_id in reliable_nodes:
        pairs_nodes, connected_nodes = tree.calcu_neighbor_btw_subtree(cur_node_id,reliable_node_id,mixin_near_matrix)
        if(len(pairs_nodes)==0):
            continue
        # return pairs_nodes
        cur_node_offspring = tree.get_subtree_offspring_id(cur_node_id,[cur_node_id])
        local_density_cur_offspring = np.mean(local_density[cur_node_offspring])
        local_density_connected_nodes = np.mean(local_density[connected_nodes])
        if(local_density_connected_nodes>local_density_cur_offspring*dc_eps):
            return pairs_nodes
        pass
    #* 2. 
    for i in range(len(not_reliable_nodes)):
        pairs_nodes, connected_nodes = tree.calcu_neighbor_btw_subtree(cur_node_id,not_reliable_nodes[i],mixin_near_matrix)
        if(len(pairs_nodes)==0):
            pairs_nodes = is_connected_new(tree,local_density,dc_eps,not_reliable_nodes[i],reliable_nodes,not_reliable_nodes[i+1:],mixin_near_matrix)
            if(len(pairs_nodes)>0):
                return pairs_nodes
        else:
            cur_node_offspring = tree.get_subtree_offspring_id(cur_node_id,[cur_node_id])
            local_density_cur_offspring = np.mean(local_density[cur_node_offspring])
            local_density_connected_nodes = np.mean(local_density[connected_nodes])
            if(local_density_connected_nodes>local_density_cur_offspring*dc_eps):
                return pairs_nodes


            # return pairs_nodes
        # #* 连通点平均密度大于局部密度阈值，则更新最大相似度
        cur_node_offspring = tree.get_subtree_offspring_id(cur_node_id,[cur_node_id])
        local_density_cur_offspring = np.mean(local_density[cur_node_offspring])
        local_density_connected_nodes = np.mean(local_density[connected_nodes])
        if(local_density_connected_nodes>local_density_cur_offspring*dc_eps):
            return pairs_nodes
        if(len(pairs_nodes)==0):
            pairs_nodes = is_connected_new(tree,local_density,dc_eps,not_reliable_nodes[i],reliable_nodes,not_reliable_nodes[i+1:],mixin_near_matrix)
            if(len(pairs_nodes)>0):
                return pairs_nodes
        # pass
    return []


def label_these_node_new(outlier_forest,cluster_forest,node_num,uncertain_forest,mixin_near_matrix):
    '''
        给森林中的样本点贴标签
        考虑不确定点的分配
    '''
    labels = np.full((node_num),-1,dtype=np.int32)
    for outlier_id in outlier_forest:
        outlier_tree = outlier_forest[outlier_id]
        outlier_idlist = outlier_tree.get_subtree_offspring_id(outlier_id,[outlier_id])
        labels[outlier_idlist] = -1
        pass
    
    label = 0
    for tree_id in cluster_forest:
        cluster_tree = cluster_forest[tree_id]
        cluster_idlist = cluster_tree.get_subtree_offspring_id(tree_id,[tree_id])
        labels[cluster_idlist] = label
        label = label + 1
        pass

    #todo 修改此处代码
    for uncertain_tree_id in uncertain_forest:
        uncertain_tree = uncertain_forest[uncertain_tree_id]
        uncertain_nodes_id = uncertain_tree.get_subtree_offspring_id(uncertain_tree_id,[uncertain_tree_id])
        all_near_nodes = np.array([],dtype=np.int32)
        for node_id in uncertain_nodes_id:
            all_near_nodes = np.append(all_near_nodes,mixin_near_matrix[node_id])
            pass
        # all_near_nodes = mixin_near_matrix[uncertain_nodes_id]
        all_near_nodes = np.unique(all_near_nodes)
        all_near_nodes = all_near_nodes[np.where(labels[all_near_nodes]!=-1)]
        unique_labels,counts=np.unique(labels[all_near_nodes],return_counts=True)
        if(len(counts)==0):
            cur_label = -1
        else:
            cur_label = unique_labels[np.argmax(counts)]
        labels[uncertain_nodes_id]=cur_label
        pass

    core_points = cluster_forest.keys()
    return labels,core_points



'''
密度峰值树；
根据cfsfdp算法生成的局部密度、高密度最近邻距离、决策指标来生成 DPTree；
'''
class Node():
    def __init__(self,node_id,attr_list,parent_id=None,dist_to_parent=None,density=None,gamma=None,children=[]):
        self.node_id = node_id
        self.attr_list = attr_list
        self.parent_id = parent_id
        self.dist_to_parent = dist_to_parent
        self.density = density
        self.children = children
        self.gamma = gamma
        self.offspring_num = None
        self.lvl = None

    def addChild(self,child):
        self.children+=[child]

    def removeChild(self,child):
        self.children.remove(child)
    
    def resetChildren(self):
        self.children = []

    def setParentId(self,parent_id):
        self.parent_id = parent_id

    def setOffspringNum(self,num):
        self.offspring_num = num

    def setLvl(self,lvl):
        self.lvl = lvl

    def getAttr(self):
        return self.attr_list

    def getNodeId(self):
        return self.node_id

    def getParentId(self):
        return self.parent_id
    
    def getDistToParent(self):
        return self.dist_to_parent
    
    def getDensity(self):
        return self.density

    def getGamma(self):
        return self.gamma

    def getChildren(self):
        return self.children
    
    def hasChildren(self,child_id):
        if child_id in self.children:
            return True
        else:
            return False

    def getOffspringNum(self):
        return self.offspring_num

    def getLvl(self):
        return self.lvl




class DPTree():
    def __init__(self):
        self.node_count = 0
        self.node_dir = {}
        self.root_node = None
        self.node_offspring = {}
        self.sorted_gamma_index = None
        pass

    def createTree(self,X,sorted_gamma_index,closest_node_id,closest_dis_denser,local_density,gamma):
        #* 根据 gamma 顺序新建节点
        node_dir = {}
        node_created = np.zeros(len(sorted_gamma_index))
        self.sorted_gamma_index = sorted_gamma_index
        for i in range(len(sorted_gamma_index)):
            node_id = sorted_gamma_index[i]
            parent_id = closest_node_id[node_id] #* closest_node_id是根据排序后的gamma获得的
            attr_list = X[node_id]
            dist_to_parent = closest_dis_denser[node_id]
            density = local_density[node_id]
            if(node_created[node_id]==0):
                node = Node(node_id,attr_list,parent_id,dist_to_parent=dist_to_parent,density=density,gamma[node_id],children=[])
                node_created[node_id] = 1
                node_dir[node_id] = node
            node_dir[node_id].setParentId(parent_id)
            if(node_created[parent_id]==0):
                parent_node = Node(parent_id,X[parent_id],parent_id=None,dist_to_parent=closest_dis_denser[parent_id],density=local_density[parent_id],gamma=gamma[parent_id],children=[])
                node_created[parent_id] = 1
                node_dir[parent_id] = parent_node
            parent_node = node_dir[parent_id]
            cur_node = node_dir[node_id]
            if(node_id != parent_id):#* 非根节点
                parent_node.addChild(node_id)
                # parent_lvl = parent_node.getLvl()
                # cur_node.setLvl(parent_lvl+1)
            else:
                if(parent_node.getLvl()==None):
                    parent_node.setLvl(0)

        #* 设置节点层次信息
        # for i in tree.node_dir:

        #     pass
                
        self.root_node = node_dir[sorted_gamma_index[0]]
        self.node_dir = node_dir
        self.node_count = len(sorted_gamma_index)
        pass

    def printTree2(self,parent_id,spaceStr=''):
        for node_id in self.node_dir:
            if(node_id==self.root_node.node_id):
                continue
            node = self.node_dir[node_id]
            if(node.parent_id==parent_id):
                print(spaceStr, node.node_id, sep = '')
                self.printTree2(node.node_id,spaceStr+'     ')
        pass
    
    def calcu_subtree_offspring_num(self,node_id):
        node = self.node_dir[node_id]
        cur_offsprings = node.getOffspringNum()
        if(cur_offsprings!=None):
            return cur_offsprings
        child_num = len(node.children)
        if(child_num==0):
            return 0
        for i in node.children:
            cur_offsprings = self.calcu_subtree_offspring_num(i)
            child_num+=cur_offsprings
        node.setOffspringNum(child_num)
        return child_num

    def get_subtree_offspring_id(self,node_id,other_idlist):
        '''
            获取所有子孙的node_id
            考虑：是否需要存储在node属性中。
        '''
        def fn_get_subtree_offspring_id(node_id,offspring_idlist):
            if(node_id in self.node_offspring.keys()):
                return self.node_offspring[node_id]
            else:
                node = self.node_dir[node_id]
                children = node.getChildren()
                child_num = len(children)
                if(child_num==0):
                    self.node_offspring[node_id] = offspring_idlist
                    return offspring_idlist
                offspring_idlist= list(offspring_idlist) + children
                for i in children:
                    child_offspring_idlist = fn_get_subtree_offspring_id(i,[])
                    self.node_offspring[i] = child_offspring_idlist
                    offspring_idlist= list(offspring_idlist) + child_offspring_idlist
                    pass
                self.node_offspring[node_id] = offspring_idlist
                return offspring_idlist             
        offspring_idlist = fn_get_subtree_offspring_id(node_id,[])
        return np.array(list(offspring_idlist) + other_idlist)
        
        

    def calcu_subtree_entropy(self,offspring_id,local_density,closest_dis_denser):
        p_sum = np.sum(local_density[offspring_id]/closest_dis_denser[offspring_id])
        p = (local_density[offspring_id]/closest_dis_denser[offspring_id])/p_sum
        entropy = -1*np.sum(p*np.log2(p))
        #* 只有一个点的情况返回 0
        if(entropy==0):
            return 0
        return entropy/(-1*np.log2(1/(len(offspring_id))))

    
    def remove_subtree(self,child_id):
        '''
            删除 node_id 节点的子树：child_id, 被删除的子树形成新的树并返回
            1. 更新 self.node_dir, self.node_count
            2. 更新 node_id 节点的 children[], 以及所有父级offspring_num
            3. 生成新树
        '''
        # print("删除子节点：",child_id)
        offspring_id = self.get_subtree_offspring_id(child_id,[child_id])
        offspring_len = len(offspring_id)
        node_id = self.node_dir[child_id].parent_id
        node = self.node_dir[node_id]
        node.removeChild(child_id)
        self.node_count = self.node_count-offspring_len
        #* 删除存储的子孙节点
        if(node_id in self.node_offspring.keys()):
            for node_to_delete in offspring_id:
                self.node_offspring[node_id].remove(node_to_delete)
                print("删除子孙节点:",node_to_delete)
                pass
            pass
        # cur_id = child_id
        # parent_id = node_id
        # #* 设置父级 offspring_num:
        # while(cur_id!=parent_id):
        #     parent_node = self.node_dir[parent_id]
        #     if(parent_node.getOffspringNum()!=None):
        #         parent_node.setOffspringNum(parent_node.getOffspringNum()-offspring_len)
        #     cur_id = parent_id
        #     parent_id = parent_node.parent_id
        #     pass
        #* 更新 self.node_dir, 生成新树:
        new_tree = DPTree()
        for i in offspring_id:
            removed_node = self.node_dir.pop(i)
            new_tree.node_dir[i] = removed_node
            pass
        new_tree.node_count = offspring_len
        new_tree.root_node = new_tree.node_dir[child_id]
        new_tree.root_node.setParentId(child_id)
        return new_tree

    def calcu_dist_betw_subtree(self,node_id_one,node_id_two,dist_mat,eps):
        '''
            计算两个子树间的连通距离
            return：
                1. 最短距离
                2. 小于距离阈值的点集
        '''
        connected_nodes = np.array([],dtype=np.int32)
        offspring_one = self.get_subtree_offspring_id(node_id_one,[node_id_one])
        offspring_two = self.get_subtree_offspring_id(node_id_two,[node_id_two])
        dist = float('inf')
        for i in offspring_two:
            tmp_dist = np.min(dist_mat[i][offspring_one])
            if(tmp_dist<dist):
                dist = tmp_dist
                pass
            connected_nodes_index = np.where(dist_mat[i][offspring_one]<eps)[0]
            if len(connected_nodes_index)>0:
                connected_nodes = np.r_[[i],connected_nodes,offspring_one[connected_nodes_index]]
                pass
        return dist, np.unique(connected_nodes)

    def calcu_neighbor_btw_subtree(self,node_id_one,node_id_two,mixin_near_matrix):
        '''
            计算两个子树间的邻近点
            return:
                邻近的点对
                所有邻近点
        '''
        connected_nodes = np.array([],dtype=np.int32)
        offspring_one = self.get_subtree_offspring_id(node_id_one,[node_id_one])
        offspring_two = self.get_subtree_offspring_id(node_id_two,[node_id_two])
        pairs_nodes = []
        for i in offspring_two:
            connected_nodes_index = np.intersect1d(mixin_near_matrix[i],offspring_one)
            if len(connected_nodes_index)>0:
                for j in connected_nodes_index:
                    pairs_nodes.append([i,j])
                    pass
                pass
        if(len(pairs_nodes)==0):
            return pairs_nodes,connected_nodes
        return np.array(pairs_nodes), np.unique(np.array(pairs_nodes).flatten())


    def calcu_dist_betw_subtree_entropy(self,node_id_one,node_id_two,dist_mat,eps):
        '''
            计算两个子树间的连通距离
            return：
                1. 最大相似距离
                2. 大于相似距离阈值的点集
        '''
        connected_nodes = np.array([],dtype=np.int32)
        offspring_one = self.get_subtree_offspring_id(node_id_one,[node_id_one])
        offspring_two = self.get_subtree_offspring_id(node_id_two,[node_id_two])
        dist = -1
        for i in offspring_two:
            tmp_dist = np.max(dist_mat[i][offspring_one])
            if(tmp_dist>=dist):
                dist = tmp_dist
                pass
            connected_nodes_index = np.where(dist_mat[i][offspring_one]>=eps)[0]
            if len(connected_nodes_index)>0:
                connected_nodes = np.r_[[i],connected_nodes,offspring_one[connected_nodes_index]]
                pass
        return dist, np.unique(connected_nodes)


    def calcu_depth(self,node_id, depth):
        node = self.node_dir[node_id]
        parent_id = node.parent_id
        if(node_id==parent_id):
            return depth
        else:
            return self.calcu_depth(parent_id,depth+1)
