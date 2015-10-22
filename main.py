import numpy as np
from utils import *
from all import *

if __name__ == "__main__":
	n = 7;
	MP = Matching_problem(n,n)
	MP.set_random_preferences();
	MP.print_preferences()
	SM = MP.find_stable_matching()
	print SM
	print MP.generalized_stable_matching([n-1 for i in range(n)])	
	print MP.generalized_stable_matching([0 for i in range(n)])	
	SM = MP.find_stable_matching("women")
	print SM

	print "==================="

	A = find_all_possible_partners(MP)

	print A
	SM = find_all_stable_matchings(A,MP)
	print SM
