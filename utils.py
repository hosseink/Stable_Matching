import sys
import os
import math
import numpy as np
import random
import time
import collections
import copy
import csv
import matplotlib.pyplot as plt

def index_to_value(arr):
    if len(arr.shape) == 1:
        arr = np.array([arr])
    Q = np.zeros(arr.shape, dtype = np.int8)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            Q[i, arr[i,j]] = j
    if len(arr.shape) == 1:
        return Q[0,:]
    return Q

def isConsistence(match1, match2):
	print match1, match2
	if len(match1) != len(match2):
		return False;        
	
	
	for k in match1:
		if match1[k] not in match2:
			return False;
		if match2[match1[k]] != k:
			return False
	return True

class Preference:
    def __init__(self, n): # n is the size of the list
        self.size_of_list = n
        self.plist = np.zeros(n , dtype = np.int8)

    #################################################################################
    # sample_preference samples P from the a distributions defined by ptype:        #
    #   uniform: uniform distribution on set of all ranking lists.                  #
    #   bistochastic0: params should be a distribution on {0,..., n} which is		# 
	#				   the  probability of the placement of first memeber.			#
    #   distModel: params should be a set of sampler functions  from [0,1]          #
    #################################################################################
    def sample_preference(self, ptype = "uniform", params = None):

		if ptype == "uniform":
			self.plist = np.random.permutation(self.size_of_list)

		elif ptype == "bistochastic0":
			if len(params) < self.size_of_list:
				print "input params is not valid"
				raise

			self.plist = np.random.permutation(self.size_of_list)
			CMF = np.cumsum(np.array(params));
			rnd = random.random()
			idx = [e[0] for e in enumerate(CMF) if e[1] > rnd][0]
			previdx = [e[0] for e in enumerate(self.plist) if e[1]==0][0]
			tmpval = self.plist[idx];
			self.plist[idx] = 0
			self.plist[previdx] = tmpval

		elif ptype == "distModel":
			if params == None:
				params = []
			if len(params) < self.size_of_list:
				params.extend([lambda : random.random() for _ in range(self.size_of_list - len(params))])
				print params
				samples = [params[i]() for i in range(self.size_of_list)];
				self.plist = np.array([t[0] for t in sorted([i for i in enumerate(samples)], key = lambda x: x[1])])
					
		else:
			print "ptype is not valid"
			raise

class Preference_list:
	def __init__(self, n, m): # m preference list of size n
		self.num_of_lists = n
		self.size_of_list = m
		self.plist = np.zeros([n, m], dtype = np.int8) 
		self.plist_ = np.zeros([n, m], dtype = np.int8) 

	def sample_preferences(self, ptype = 'uniform', params = None):
		for i in range(self.num_of_lists):
			P = Preference(self.size_of_list);
			P.sample_preference(ptype, params);
			self.plist[i,:] = P.plist
		self.plist_ = index_to_value(self.plist)	
	def print_preferences(self, pFormat = "plist"):
		if pFormat == "plist":
			print '\n'.join([' '.join(['%2d' %p for p in pref]) for pref in self.plist])
		else:
			print '\n'.join([' '.join(['%2d' %p for p in pref]) for pref in self.plist_])
	


class Matching:
	def __init__(self, n,m, men_match = {}, women_match = {}):
		self.num_of_men = n
		self.num_of_women = m
		self.men_match = men_match
		self.women_match = women_match
		self.update()

	def update(self):
		if len(self.men_match)==0:
			self.men_match = {self.women_match[k]:k for k in self.women_match}

		if len(self.women_match)==0:
			self.women_match = {self.men_match[k]:k for k in self.men_match}

		assert(isConsistence(self.men_match, self.women_match))		
	
	def is_complete_matching(self):
		return len(self.men_match)==min(self.num_of_men, self.num_of_women)

	def is_stable_matching(self, men_preference, women_preference):
		if not self.is_complete_matching():
			return False;
		for man in range(self.num_of_men):
			if man not in self.men_match:
				for woman_i in range(self.num_of_women):
					man_i = self.women_match[woman];
					if women_preference_plist_[woman_i, man] < \
							women_preference_plist_[woman_i, man_i]:
						return False;
	 	
			else:
				woman = self.man_match[man]
				r = 0;
				woman_r = men_preference.plist[man, r]
				while woman_r != woman:
					if woman_r not in self.women_match:
						return False
					man_r = women_match[woman_r]
					if women_preference_plist_[woman_r, man] < \
							women_preference_plist_[woman_r, man_r]:
						return False
					r += 1
					woman_r = men_preference.plist[man, r]
		return True;	

	def rank_representation(self, men_preference, women_preference):
		men_match = {man: men_preference.plist_[man, self.men_match[man]] 
							for man in self.men_match}
		women_match = {woman: women_preference.plist_[woman, self.women_match[woman]] 
							for woman in self.women_match}
		return {"men_match": men_match, "women_match": women_match}
	
	def plot(self, men_preference = None, women_preference = None):

		n = self.num_of_men
		m = self.num_of_women
		maxn = max(n,m)
		s = min(1, 10./maxn);	
		print s
		ax=plt.figure(figsize = (4, (maxn + 2)  * s)).add_subplot(111)

		def addconnection(i,j,c):
		  return [(-1,1),(s*maxn-i*s,s*maxn-s*j),c]

		def drawnodes(names,i):
		  if(i==1):
			color='r'
			posx=-1
		  else:
			color='b'
			posx=1

		  posy=maxn*s
		  for n in names:
			plt.gca().add_patch( plt.Circle((posx,posy),radius=0.05,fc=color))
			if posx==1:
			  ax.annotate(n,xy=(posx,posy+0.1))
			else:
			  ax.annotate(n,xy=(posx-len(n)*0.1,posy+0.1))
			posy-=s

		men_set=['M'+str(i) for i in range(n)]
		women_set=['W'+str(i) for i in range(m)]
		if (men_preference is not None and women_preference is not None):
			men_set=['M'+str(man) + "(" + 
						str(men_preference.plist_[man , self.men_match[man]]) + ")" 
						for man in range(n)]
			women_set=['W'+str(woman) + "(" + 
						str(women_preference.plist_[woman , self.women_match[woman]]) +")" 
						for woman in range(m)]
			
		plt.axis([-2,2,0,maxn * s + s])
		frame=plt.gca()
		frame.axes.get_xaxis().set_ticks([])
		frame.axes.get_yaxis().set_ticks([])

		drawnodes(men_set,1)
		drawnodes(women_set,2)

		connections=[addconnection(man, self.men_match[man], 'black') for man in self.men_match]
				
		for c in connections:
			print c
			plt.plot(c[0],c[1],c[2])

		plt.show()
				




