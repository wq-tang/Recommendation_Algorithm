import numpy as np
import pandas as pd 
from functools import reduce
from operator import itemgetter
from userCF import UserCF
class ItemCF(UserCF):
	"""docstring for ItemCF apply to dataset of moivelens"""
	def __init__(self,filePath=None,k=5):
		super().__init__(filePath=filePath,k=k)

	def itmeSimilarity(self):
		user_item = self.train
		item_user = self.__reverse_iu(self.train)
		w = {u:{} for u in item_user.keys()}

		log_dict = {}
		for user,item in user_item.items():#可以的话，那种热门的人类应该被去除。否则至少也要加上软惩罚
			if user in log_dict:
				log_dict[user]+= 1/np.log(1+len(item))
			else:
				log_dict[user] = 1/np.log(1+len(item))

		for user,items in user_item.items():
			for itemi in items:
				for itemj in items:
					if itemi==itemj:
						continue
					else:
						if itemj in w[itemi]:
							w[itemi][itemj]+=log_dict[user]
						else:
							w[itemi][itemj]=log_dict[user]

		for itemi in item_user.keys():
			for itemj in w[itemi].keys():
				w[itemi][itemj] /= np.sqrt(len(item_user[itemi])*len(item_user[itemj]))

		for itemi in item_user.keys(): #归一化
			maxs = max(w[itemi].values())
			for itemj in w[itemi].keys():
				w[itemi][itemj] /= maxs

		for itemu , dicts_item in w.items():
			tuples = sorted(dicts_item.items(),key = itemgetter(1),reverse=True)[:self.k]
			w[itemu] = {i:pi for i,pi in tuples}
		self.w = w

	def recommend(self,commend_num=10):
		rank={}
		for user in self.test.keys():
			local_rank = {}
			favorite_items = self.train[user]

			for item in favorite_items:
				for it,pi in self.w[item].items():
					if it in self.train[user]:
						continue
					else:
						if it in local_rank:
							local_rank[it] += pi
						else:
							local_rank[it] = pi 
			rank[user] = local_rank
		self.rank = rank
		return self.pick(commend_num)


	def __reverse_iu(self,dicts):
		reverse_out = {}
		for before,after in dicts.items():
			for af in after:
				if af not in reverse_out:
					reverse_out[af] = [before]
				else:
					reverse_out[af].append(before)
		return reverse_out


if __name__=='__main__':
	filePath = r"ml-1m\moiveLens.csv"
	itemCF = ItemCF(filePath=filePath,k=5)
	itemCF.split_data(0)
	itemCF.itmeSimilarity()
	prediction = itemCF.recommend()
	itemCF.precision(prediction)
	itemCF.recall(prediction)
	itemCF.coverage(prediction)
	itemCF.popularity(prediction)
