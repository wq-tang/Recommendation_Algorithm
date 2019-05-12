import numpy as np
import pandas as pd 
from functools import reduce
from operator import itemgetter
from userCF import UserCF
import time
#numpy.random.choice 很有用
class LFM(UserCF):
	"""docstring for LFM"""
	def __init__(self, filePath,iter_num,hid_cla,learning_rate,lambdas,commend_num=10):
		super().__init__(filePath=filePath,commend_num=commend_num,k=0)
		self.iter_num=iter_num
		self.hid_cla=hid_cla
		self.learning_rate=learning_rate
		self.lambdas=lambdas
	
	def __RandomSelectNegativeSample(self):
		#计算正样本
		self.train_data  = {}
		for user,items in self.train.items():
			user_data = {}
			for item in items:
				user_data[item] = 1
			self.train_data[user] = user_data
		#计算流行度
		self.__itemPopularity()
		sums = sum(self.item_popularity.values())
		posibility = np.array(list(self.item_popularity.values()))/sums

		#采样负样本
		for user,item in self.train.items():
			num = len(item)
			while num>0:
				sample = set(np.random.choice(list(self.item_popularity.keys()),num,True,posibility))
				original = set(self.train_data[user].keys())
				new = sample - original
				if num-len(new)>=0:
					num -= len(new)
					for item in new:
						self.train_data[user][item] = 0
				else:
					for i in range(num):
						self.train_data[user][new.pop()] = 0
					break


	def __trainLFM(self):
		for step in range(self.iter_num):
			for user,items in self.train_data.items():
				for item , rui in items.items():
					eui = rui - self.__predict(user,item)
					for k in range(self.hid_cla):  #可以考虑多批次迭代的方法
						self.P[user][k] += self.learning_rate*(eui*self.Q[k][item] - self.lambdas*self.P[user][k])
						self.Q[k][item] += self.learning_rate*(eui*self.P[user][k] - self.lambdas*self.Q[k][item])
			self.learning_rate *= 0.9

	def __predict(self,user,item):
		pui = 0
		for i in range(self.hid_cla):
			pui += self.P[user][i]*self.Q[i][item]
		return pui

	def __recomend(self):
		rank = {}
		for user in self.test.keys():
			local_rank={}
			for i,pui in enumerate(self.Q[user]):
				for k,pik in self.P[i].items():
					if k not in local_rank:
						local_rank[k] += pui*pik
					else:
						local_rank[k] = pui*pik
			rank[user] = local_rank
		self.rank = rank
		return self.pick()


	def __initModel(self):
		P = {}
		Q = [{} for i in range(len(self.hid_cla))]
		for user in self.train_data.keys():
			P[user] = np.random.random(len(self.hid_cla))

		for items in self.train_data.values():
			for item in items:
				if item not in Q[0]:
					for q in Q:
						q[item] = np.random.random()

		self.P = P
		self.Q = Q

	
	def __itemPopularity(self):
		item_popularity = {}
		for user,items in self.train.items():
			for item in items:
				if item in item_popularity:
					item_popularity[item]+=1
				else:
					item_popularity[item] = 1
		self.item_popularity = item_popularity

	def calls(self,split_point = 0):
		t0 = time.time()
		self.split_data(split_point)
		t1 = time.time()
		print("split:",t1-t0)
		self.__RandomSelectNegativeSample()
		t2 = time.time()
		print("randomSelect:",t2-t1)
		self.__initModel()
		t3= time.time()
		print("init:",t3-t2)

	def call(self):
		t1 = time.time()
		self.__trainLFM()
		t2 = time.time()
		print("train:",t2-t1)

		prediction = self.__recommend()
		print("recommend:",time.time()-t2)

		print("precision:",self.precision(prediction))
		print("recall:",self.recall(prediction))
		print("coverage:",self.coverage(prediction))
		print("popularity:",self.popularity(prediction))
		return prediction

if __name__=='__main__':
	lfm = LFM(filePath=r"ml-1m\moiveLens.csv",iter_num = 100,hid_cla=100,learning_rate=0.02,lambdas=0.01)
	lfm.calls(split_point=0)
	prediction = lfm.call()