import time
import numpy as np
import pandas as pd 
from functools import reduce
from operator import itemgetter
from userCF import UserCF
class PR(UserCF):
    """docstring for PR"""
    def __init__(self,alpha,max_depth,filePath=None,commend_num=10):
        super().__init__(filePath=filePath,commend_num=commend_num)
        self.alpha =alpha
        self.max_depth = max_depth



    def PersonalRank(self,root):
        rank=dict()
        rank={x:0 for x in self.G.keys()}
        rank[root]=1
        #开始迭代
        begin=time.time()
        for k in range(self.max_depth):
            tmp={}
            for user,items in self.train.items():
                length = len(items)
                for item in items:
                    tmp[item]+=self.alpha*rank[user]/length
            tmp[root]+=(1-self.alpha)
            rank=tmp
        end=time.time()
        print ('use_time',end-begin)
        lst=sorted(rank.items(),key=lambda x:x[1],reverse=True)
        for ele in lst:
            print("%s:%.3f, \t" %(ele[0],ele[1]))
        return rank

    def recommend(self):
        rank = {}
        self.item_user = self.reverse_iu(self.train)
        for user in self.test.keys():
            local_rank = self.PersonalRank(user)
            rank[user] = self.__get(local_rank)
        return self.pick()
    def __get(self,local_rank):
        ranks = {}
        for item in self.item_user.keys():
            ranks[item] = local_rank[item]
        return ranks


if __name__ == '__main__':
    alpha = 0.8
    max_depth = 20
    pr = PR(filePath=r"ml-1m\moiveLens.csv",alpha=alpha,max_depth=max_depth)
    prediction=pr.recommend()
    print("precision:",pr.precision(prediction))
    print("recall:",pr.recall(prediction))
    print("coverage:",pr.coverage(prediction))
    print("popularity:",pr.popularity(prediction))