import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sets import Set
                
def find_all_possible_partners(MP):
    n = MP.num_of_men
    possible_partners = {i:[] for i in range(n)}
    for w in range(n):
        for i in range(n):
            klist = [n for _ in range(n)];
            klist[w] = i;
            sm = MP.generalized_stable_matching(klist)
            if sm[w] not in possible_partners[w]:
                possible_partners[w].append(sm[w])
    return possible_partners

def merg_matchings(m1, m2, MP):
    if len(Set(m1.values()).intersection(Set(m2.values()))) > 0:
        return None
    for w1 in m1.keys():
        for w2 in m2.keys():
            if MP.men_preference[m1[w1]][w2] < MP.men_preference[m1[w1]][w1] and \
                     MP.women_preference[w2][m1[w1]] < MP.women_preference[w2][m2[w2]]:
                return None
            if MP.men_preference[m2[w2]][w1] < MP.men_preference[m2[w2]][w2] and \
                     MP.women_preference[w1][m2[w2]] < MP.women_preference[w1][m1[w1]]:
                return None
    output = m1.copy()
    output.update(m2)
    return output

def find_all_stable_matchings(possible_partners, MP):
    n = len(possible_partners)
    if n==1:
        w,men = possible_partners.items()[0]
        all_matchings = [{w:m} for m in men]
        return all_matchings

    pp1 = dict(possible_partners.items()[:n/2])
    pp2 = dict(possible_partners.items()[n/2:])
    m1_set = find_all_stable_matchings(pp1, MP)
    m2_set = find_all_stable_matchings(pp2, MP)
    all_matchings = []
    for m1 in m1_set:
        for m2 in m2_set:
            new_matching = merg_matchings(m1,m2, MP)
            if new_matching is not None:
                all_matchings.append(new_matching)

    return all_matchings
    
def matching_to_seq(m, MP):
    return [MP.women_preference[i][m[i]] for i in range(len(m))]

def plot_all(MR):
    n = len(MR[0])
    for mr in MR:
        mr = [i+1 for i in mr]
        #mr = sorted(mr)
        plt.scatter(range(1,n+1), [n - x for x in mr], color = 'black') 
        plt.plot(range(1,n+1), [n - x for x in  mr]) 

    plt.show()

if __name__ == "__main__":
    """
    n = 20
    MP = matching_problem(n,n)
    MP.set_random_preferences()
    MP.print_preferences()
    A = find_all_stable_matchings(MP)
    print A
    """
    n = 8
    Mr = np.array( [[3, 1, 5, 7, 4, 2, 8, 6],
                    [6, 1, 3, 4, 8, 7, 5, 2],
                    [7, 4, 3, 6, 5, 1, 2, 8],
                    [5, 3, 8, 2, 6, 1, 4, 7],
                    [4, 1, 2, 8, 7, 3, 6, 5],
                    [6, 2, 5, 7, 8, 4, 3, 1],
                    [7, 8, 1, 6, 2, 3, 4, 5],
                    [2, 6, 7, 1, 8, 3, 4, 5] ])
    Wr = np.array( [[4, 3, 8, 1, 2, 5, 7, 6],
                    [3, 7, 5, 8, 6, 4, 1, 2],
                    [7, 5, 8, 3, 6, 2, 1, 4],
                    [6, 4, 2, 7, 3, 1, 5, 8],
                    [8, 7, 1, 5, 6, 4, 3, 2],
                    [5, 4, 7, 6, 2, 8, 3, 1],
                    [1, 4, 5, 6, 2, 8, 3, 7],
                    [2, 5, 4, 3, 7, 8, 1, 6] ])

    Mrr = np.zeros([8,8])
    for i in range(8):
        for j in range(8):
            Mrr[i,Mr[i,j]-1] =j + 1
    
    Wrr = np.zeros([8,8])
    for i in range(8):
        for j in range(8):
            Wrr[i,Wr[i,j]-1] = j + 1
    print Wrr
    print Mrr
    MP = matching_problem(n,n)
    MP.set_preferences('men', Mrr)
    MP.set_preferences('women', Wrr)
    
    n = 5
    flag = True
   # while flag:

    A = np.array([[ 3,  2,  1,  4,  0],
                 [4,  0,  3,  2,  1],
                 [1,  4,  3,  0,  2],
                 [1,  2,  3,  0,  4],
                 [3,  2,  4,  1,  0]])
    B = np.array([[2,  2,  4,  0,  3],
                 [0,  4,  3,  1,  4],
                 [4,  1,  1,  3,  1],
                 [3,  0,  2,  4,  0],
                 [1,  3,  0,  2,  2]])
    
    MP = matching_problem(n,n)
    #MP.set_random_preferences()
    MP.set_preferences('men', A)
    MP.set_preferences('women',B.T)
    MP.print_preferences()

    A = find_all_possible_partners(MP)
    
    print A
    SM = find_all_stable_matchings(A,MP)
    print SM
    
    print len(SM)
    print SM[0]
    MR = [matching_to_seq(m,MP) for m in SM]
    if len(SM) > 4:
        flag = False
    plot_all(MR)

    
    
    
