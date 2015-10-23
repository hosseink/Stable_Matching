import sys
import os
import math
import numpy as np
import random
import time
import collections
import copy
import csv

class Matching_problem:

	def __init__(self, num_of_men, num_of_women):
		
		self.num_of_men = num_of_men;
		self.num_of_women = num_of_women;
		self.men_preference = None
		self.men_preference_ = None
		self.women_preference = None
		self.women_preference_ = None
		 
		
	def set_random_preferences(self, ptype = "uniform", params = None):
		men_plists = Preference_lists(self.num_of_men, self.num_of_women)
		men_plists.sample_preferences(ptype, params)
	
		women_plists = Preference_lists(self.num_of_women, self.num_of_men)
		women_plists.sample_preferences(ptype, params)

		self.men_preference = men_plists.plist
		self.men_preference_ = index_to_value(men_plists.plist)
	
		self.women_preference = women_plists.plist
		self.women_preference_ = index_to_value(women_plists.plist)

	def print_preferences(self, matching=None, who=None):
		if matching==None:
			if(who=="men"):
				print "men preferences: \n", '\n'.join([' '.join(['%2d' %p for p in pref]) for pref in self.men_preference])
			elif(who=="women"):
				print "women preferences: \n", '\n'.join([' '.join(['%2d' %self.women_preference[woman][man] \
					for woman in range(self.num_of_women)]) for man in range(self.num_of_men)])
			else:
				print "men preferences: \n", '\n'.join([' '.join(['%2d' %p for p in pref]) for pref in self.men_preference])
				print "women preferences: \n", '\n'.join([' '.join(['%2d' %self.women_preference[woman][man] \
					for woman in range(self.num_of_women)]) for man in range(self.num_of_men)])
		else:
			men_preference = [['  ' for i in range(self.num_of_women)] for j in range(self.num_of_men)]; 
			women_preference = [['  ' for i in range(self.num_of_men)] for j in range(self.num_of_women)];
			men_match = matching["men"];
			for key in men_match.keys():
				men_preference[key][self.men_preference_[key, men_match[key]]] = '%2d' %men_match[key];
				women_preference[men_match[key]][self.women_preference_[men_match[key],key]] = '%2d' %key;
			if(who=="men"):
				print "men preferences: \n", '\n'.join([' '.join([p for p in pref]) for pref in men_preference])
			elif(who=="women"):
				print "women preferences: \n", '\n'.join([' '.join([women_preference[woman][man] \
					for woman in range(self.num_of_women)]) for man in range(self.num_of_men)])
			else:
				print "men matching: \n", '\n'.join([' '.join([p for p in pref]) for pref in men_preference])
				print "women preferences: \n", '\n'.join([' '.join([p for p in pref]) for pref in women_preference])
			
				
	def set_preferences(self, who, preference):
		assert who in ["men", "women"];
		if who=="men":
			assert len(preference) == self.num_of_men;	
			assert len(preference[0]) == self.num_of_women
			self.men_preference = preference;
			self.men_preference_ = index_to_value(self.men_preferece)
		if who=="women":
			assert len(preference) == self.num_of_women;	
			assert len(preference[0]) == self.num_of_men
			self.women_preference = preference;
			self.women_preference_ = index_to_value(self.women_preferece)

	def is_matching(self, matching):
		matching_size = min(self.num_of_men, self.num_of_women);
		men_match = matching['men'];
		women_match = matching['women'];
		if len(men_match)!=len(women_match):
			return False;
		for man in men_match.keys():
			woman = men_match[man];
			if man!= women_match[woman]:
				return False;
		return True;
			
	
	def is_stable_matching(self, matching):
		if not self.is_matching(matching):
			return False;
		men_match = matching['men'];
		women_match = matching['women'];
		for man in range(self.num_of_men):
			for woman in range(self.num_of_women):
				if not man in men_match and not woman in women_match:
					return False;
				elif not man in men_match:
					woman_match = women_match[woman];
					if self.women_preference_[woman][woman_match] > self.women_preference_[woman][man]:
						return False;
				
				elif not woman in women_match:
					man_match = men_match[man];
					if self.men_preference_[man][man_match] > self.men_preference_[man][woman]:
						return False;

				elif not woman == men_match[man]:
					man_match = men_match[man];
					woman_match = women_match[woman];
					if (self.men_preference_[man][man_match] > self.men_preference_[man][woman] and
						self.women_preference_[woman][woman_match] > self.women_preference_[woman][man]):
							return False;
		return True;
	

	def find_stable_matching(self, optimal = "men"):
		
		n = self.num_of_men
		m = self.num_of_women
		women_ordered_list = [[self.men_preference[i,m-j-1] for j in range(m)] for i in range(n)]
		women_preference = copy.deepcopy(self.women_preference_)
		
		if optimal == "women":
			n = self.num_of_women
			m = self.num_of_men
			women_ordered_list = [[self.women_preference[i,m-j-1] for j in range(m)] for i in range(n)]
			women_preference = copy.deepcopy(self.men_preference_)
		
		men_match = {}
		women_match = {}
		unmatched_men = set(range(n))
		while(len(unmatched_men) > 0):
			man = unmatched_men.pop();
			while(True):
				woman = women_ordered_list[man].pop();
				if not woman in women_match:
					women_match[woman] = man
					men_match[man] = woman
					break;
				elif women_preference[woman, man] < women_preference[woman, women_match[woman]]:
					unmatched_men.add(women_match[woman])
					women_match[woman] = man
					men_match[man] = woman
					break;
	
		############ ******************* Add other statistic to the output	
		matching = {"men":men_match, "women": women_match}	
		if optimal == "women":
			matching = {"men":women_match, "women": men_match}	
		assert(self.is_stable_matching(matching))
		return matching

	
	def kmatching(self, klist):
		n = self.num_of_men;
		m = self.num_of_women;
		men_p = self.men_preference_;
		men_p2 = [[x[0] for x in sorted([y for y in enumerate(p)], key = lambda y:y[1])] for p in men_p]
		women_p = self.women_preference_;
		men_match = {}
		women_match = {}
		l = 0;
		proposed = [-1 for i in range(n)];
		while True:
			unmatched_men = [man for man in range(n) if man not in men_match.keys()]
			if (len(unmatched_men)==0):
				break;
			new_set_of_men = [];
			for man in unmatched_men:
				if proposed[man]<m-1:
					new_set_of_men.append(man);
			if len(new_set_of_men) == 0:
				break;
			for man in new_set_of_men:
				proposed[man] += 1;
				woman = men_p2[man][proposed[man]];
				if women_p[woman][man] > klist[woman]:
					continue;
				if woman not in women_match.keys():
					men_match[man] = woman;
					women_match[woman] = man;
				elif women_p[woman][women_match[woman]] > women_p[woman][man]:
					del(men_match[women_match[woman]]);
					men_match[man] = woman;
					women_match[woman] = man;
		return {"men":men_match, "women":women_match};
				
	def generalized_stable_matching(self, klist):
		n = self.num_of_men;
		m = self.num_of_women;
		prev_unmatched_women = range(m);
		while True:
			sm = self.kmatching(klist);
			unmatched_women = [woman for woman in range(m) if woman not in sm["women"].keys()];
			if len(unmatched_women) == 0:
				break;
			for woman in unmatched_women:
				if woman in prev_unmatched_women and klist[woman] < n-1:
					klist[woman] += 1;
			prev_unmatched_women = unmatched_women;
		#return sm
		return [self.women_preference[woman][sm['women'][woman]] for woman in range(m)];
	
	def remove_man(self, idx):
		n = self.num_of_men;
		m = self.num_of_women;
		del(self.men_preference[idx])
		for i in range(m):
			man_pref = self.women_preference[i][idx];
			del(self.women_preference[i][idx]);
			for j in range(n-1):
				if self.women_preference[i][j] > man_pref:
					self.women_preference[i][j] -= 1;
		self.num_of_men -=1;

	def remove_woman(self, idx):
		n = self.num_of_men;
		m = self.num_of_women;
		del(self.women_preference[idx])
		for i in range(n):
			woman_pref = self.men_preference[i][idx];
			del(self.men_preference[i][idx]);
			for j in range(m-1):
				if self.men_preference[i][j] > woman_pref:
					self.men_preference[i][j] -= 1;
		self.num_of_women -=1;
