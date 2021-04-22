'''
Description: 聚类前探索
Author: SongJ
Date: 2021-04-09 19:42:40
LastEditTime: 2021-04-09 19:47:30
LastEditors: SongJ
'''
from pyclustertend import hopkins, ivat, vat


#* 聚类趋势评价：hopkins
def calc_hopkins(X,sampling_size):
    tend = hopkins(X,sampling_size)
    return 1-tend


def draw_vat(X):
    vat(X)
    pass

def draw_ivat(X):
    ivat(X)
    pass
