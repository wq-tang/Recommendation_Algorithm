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
        self.split_data(0)
        self.alpha =alpha
        self.max_depth = max_depth
        self.item_user = self.reverse_iu(self.train)
        self.G = dict(self.train,**self.item_user)

    def PersonalRank(self,root):
        rank=dict()
        rank={x:0 for x in self.G.keys()}
        rank[root]=1
        #开始迭代
        begin=time.time()
        for k in range(self.max_depth):
            tmp= {x:0 for x in self.G.keys()}
            for user,items in self.train.items():
                length = len(items)
                for item in items:
                    tmp[item]+=self.alpha*rank[user]/length
                    if item == root:
                        tmp[root]+=(1-self.alpha)
            rank=tmp
        end=time.time()
        print ('use_time: ',end-begin)
        return rank

    def recommend(self):
        self.rank = {}
        for user in self.test.keys():
            local_rank = self.PersonalRank(user)
            self.rank[user] = self.__get(local_rank)
        return self.pick()
    def __get(self,local_rank):
        ranks = {}
        for item in self.item_user.keys():
            ranks[item] = local_rank[item]
        ranks=sorted(ranks.items(),key=lambda x:x[1],reverse=True)[:self.commend_num]
        return ranks

    def matrix_method(self,root):
        self.build_Matrix()
        self.rank  = {}
        for user in self.test.keys():
            local_rank={}
            start =time.time()
            r0 = np.array([0 for i in self.G.keys()]) 
            r0[self.search[root]] = 1
            rt = np.array([1 for i in self.G.keys()])
            rt /= np.sum(rt)
            rt_1 = rt+1

            while np.max(np.abs(rt-rt_1))<0.1**2:
                rt_1 = self.alpha*np.dot(self.M,rt)+(1-self.alpha)*r0

            for i,item in enumerate(self.item_user.keys()[len(self.train):]):
                local_rank[item] = rt[i]

            self.rank[user] = local_rank
            print("use_time:",time.time()-start)
        self.pick()

    def build_Matrix(self):
        search={}
        user_num = len(self.train.keys())
        M = np.array([[0 for i in range(len(self.G))] for i in range(len(self.G))])
        for i,user in enumerate(self.G.keys()):
            search[user] = i
        for i,values in enumerate(self.G.values()):
            for v in values:
                M[i][search[v]] = 1
        M = M/np.sum(M,0)
        self.search =search
        self.M = M
        

if __name__ == '__main__':
    alpha = 0.8
    max_depth = 20
    pr = PR(filePath=r"ml-1m\moiveLens.csv",alpha=alpha,max_depth=max_depth)
    prediction=pr.recommend()
    print("precision:",pr.precision(prediction))
    print("recall:",pr.recall(prediction))
    print("coverage:",pr.coverage(prediction))
    print("popularity:",pr.popularity(prediction))