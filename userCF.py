import numpy as np
import pandas as pd 
from functools import reduce
from operator import itemgetter
class UserCF(object):
	"""docstring for UserCF apply to dataset of moivelens"""
	def __init__(self,filePath=None,commend_num=10,k=5):
		super(UserCF, self).__init__()
		np.random.seed=0
		self.k = k
		self.commend_num=commend_num
		if filePath:
			self.read_data(filePath)

	def read_data(self,filePath):
		try:
			data = pd.read_csv(filePath)
			user = list(data["userId"])
			moive = list(data["moiveId"])
			self.data =list(zip(user,moive))
			np.random.shuffle(self.data) 
			self.length = len(self.data)
		except FileNotFoundError:
			print("No such file")
		else:
			print("successfully load the file")
		finally:
			print("suger:")

	def split_data(self,split_point=0):
		seq = self.length//8
		if split_point < self.length:
			if split_point == self.length-1:
				test = self.data[split_point*seq:]
				train = self.data[:split_point*seq]
			else:
				test = self.data[split_point*seq:(split_point+1)*seq]
				train = self.data[:split_point*seq]+self.data[(split_point+1)*seq:]
			self.train = {}
			self.test = {}
			for user,item in train:
				if user not in self.train:
					self.train[user] = {item}
				else:
					self.train[user].add(item)

			for user,item in test:
				if user not in self.test:
					self.test[user] = {item}
				else:
					self.test[user].add(item)					
		else:
			raise IndexError("split_point must < length")

	def recall(self,prediction):
		hit = 0
		alls = 0
		for user,items in self.test.items():
			alls+=len(items)
			hit+=len(items.intersection(prediction[user]))
		return hit/alls

	def precision(self,prediction):
		hit = 0
		alls = 0
		for user,items in self.test.items():
			alls+=len(prediction[user])
			hit+=len(items.intersection(prediction[user]))
		return hit/alls

	def coverage(self,prediction):
		label_items_set = reduce(set().union,self.train.values())
		pre_items_set = reduce(set().union,prediction.values())
		return len(pre_items_set)/len(label_items_set)

	def popularity(self,prediction):
		item_popularity={}
		for user,items in self.train.items():
			for item in items:
				if item in item_popularity:
					item_popularity[item]+=1
				else:
					item_popularity[item] = 1

		all_popularity = 0
		item_nums = 0
		for user,items in prediction.items():
			for item in items:
				all_popularity+=np.log(1+item_popularity[item])
				item_nums+=1
		return all_popularity/item_nums

	def __UserSimilarity(self):
		item_user = {}
		for user,items in self.train.items():
			for item in items:
				if item not in item_user:
					item_user[item] = [user]
				else:
					item_user[item].append(user)
		w = {u:{} for u in self.train.keys()}
		log_dict = {}
		for item,user in item_user.items():
			if item in log_dict:
				log_dict[item]+= 1/np.log(1+len(user))
			else:
				log_dict[item] = 1/np.log(1+len(user))

		for item,user in item_user.items():
			for u in user:
				for v in user:
					if u==v:
						continue
					else:
						if v in w[u]:
							w[u][v]+=log_dict[item]
						else:
							w[u][v]=log_dict[item]

		for u in self.train.keys():
			for v in w[u].keys():
				w[u][v] /= np.sqrt(len(self.train[u])*len(self.train[v]))

		# for u in self.train.keys():#归一化
		# 	maxs =w[u].values()
		# 	for v in w[u].keys():
		# 		w[u][v] /= maxs		

		self.w=w

	def __recommend(self):
		rank={}
		for user in self.test.keys():
			local_rank = {}
			w_u = self.w[user]
			w_u = sorted(w_u.items(),key = itemgetter(1),reverse = True)
			w_u = [ u for u,_ in w_u[:self.k] ]

			for v in w_u:
				for item in self.train[v]:
					if item in self.train[user]:
						continue
					else:
						if item in local_rank:
							local_rank[item]+=self.w[user][v]
						else:
							local_rank[item] = self.w[user][v]
			rank[user] = local_rank

		self.rank = rank
		return self.pick()

	def pick(self):
		prediction = {}
		for user,ranks in self.rank.items():
			tuples = sorted(ranks.items(),key = itemgetter(1),reverse = True)
			prediction[user]=[u for u,_ in tuples[:self.commend_num]]
		self.prediction = prediction
		return self.prediction

	def reverse_iu(self,dicts):
		reverse_out = {}
		for before,after in dicts.items():
			for af in after:
				if af not in reverse_out:
					reverse_out[str(af)] = [before]
				else:
					reverse_out[str(af)].append(before)
		return reverse_out

	def call(self,split_point=0):
		self.split_data(split_point)
		self.__UserSimilarity()
		prediction = self.__recommend()
		print("precision:",self.precision(prediction))
		print("recall:",self.recall(prediction))
		print("coverage:",self.coverage(prediction))
		print("popularity:",self.popularity(prediction))
		return prediction

if __name__=='__main__':
	userCF = UserCF(filePath=r"ml-1m\moiveLens.csv",k=10)
	prediction = userCF.call(split_point=0)

