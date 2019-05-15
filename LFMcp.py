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
		step=0
		max_len = len(posibility)
		lens = len(self.train.keys())
		for user,item in self.train.items():
			num = len(item)
			neg_num = min(max_len,3*num)
			sample = set(np.random.choice(list(self.item_popularity.keys()),neg_num,False,posibility))
			original = set(self.train_data[user].keys())
			new = sample - original
			if num-len(new)>=0:
				for item in new:
					self.train_data[user][item] = 0
				lenss = len(new)
			else:
				for i in range(num):
					self.train_data[user][new.pop()] = 0
				lenss = num

			print("step=%d,bili = %f, ratio=%d"%(step,step/lens,lenss/num))
			step+=1


	def __trainLFM(self):
		st = time.time()
		for step in range(self.iter_num):
			print("step = %d"%step)
			print(time.time()-st)
			for user,items in self.train_data.items():
				sigma = []
				lengths = len(items)
				for item , rui in items.items():
					sigma.append(rui - np.dot(self.P[user],self.Q[item]))
				sigma = np.array(sigma)/lengths

				for k in range(self.hid_cla):
					qk = []
					for item in items.keys():
						qk.append(self.Q[item][k])
					qk=np.array(qk)
					self.P[user][k] += self.learning_rate*(np.dot(qk,sigma) - self.lambdas*self.P[user][k])

			for item,users in self.item_user.items():
				sigma = []
				lengths = len(users)
				for user in users:
					sigma.append(self.train_data[user][item] - np.dot(self.P[user],self.Q[item]))
				sigma /= np.array(sigma)

				for k in range(self.hid_cla):
					pk = []
					for u in users:
						pk.append(self.P[u][k])
					pk = np.array(pk)
					self.Q[item][k] += self.learning_rate*(np.dot(pk,sigma) - self.lambdas*self.Q[item][k])

			self.learning_rate *= 0.9


	def __recommend(self):
		rank = {}
		for user in self.test.keys():
			local_rank={}
			for item in self.item_user.keys():
				local_rank[item] = np.dot(self.P[user],self.Q[item])
			rank[user] = local_rank
		self.rank = rank
		return self.pick()


	def __initModel(self):
		self.item_user = self.reverse_iu(self.train_data)
		self.original_P = [np.random.random(self.hid_cla) for user in range(len(self.train_data))]
		self.original_Q = [np.random.random(self.hid_cla) for user in range(len(self.item_user))]
		
		self.P = {}
		self.Q = {}
		for i,user in enumerate(self.train_data.keys()):
			self.P[user] = self.original_P[i]

		for i,item in enumerate(self.item_user.keys()):
			self.Q[item] = self.original_Q[i]



	
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
		self.__trainLFM()

	def call(self):
		t2 = time.time()
		prediction = self.__recommend()
		print("recommend:",time.time()-t2)
		print("precision:",self.precision(prediction))
		print("recall:",self.recall(prediction))
		print("coverage:",self.coverage(prediction))
		print("popularity:",self.popularity(prediction))
		return prediction

if __name__=='__main__':
	lfm = LFM(filePath=r"moiveLens.csv",iter_num = 100,hid_cla=100,learning_rate=0.02,lambdas=0.01)
	lfm.calls(split_point=0)
	prediction = lfm.call()