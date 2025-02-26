import matplotlib

import pylab as pl
import numpy as np
import copy

from arsenal.maths.checkgrad import fdcheck
from arsenal.maths.stepsize import adam
from arsenal.maths import softmax, onehot, sample
#from collections import defaultdict
from arsenal import iterview, Alphabet
from arsenal import viz
from arsenal.viz import axman

from leach.qlearn import QLearner
from leach.mdp import DiscountedMDP
from leach.gridworld import GridWorld
from leach.policy import *
from leach.teachers import *

def main():
    G = GridWorld(mapname="3x6.txt")
    [values, Si, _] = G.encode() # [MDP, State-lookups, Action-lookups]
    M = DiscountedMDP(*values, gamma=0.7)
    D = M.S * M.A + M.S + M.A

    alphabet = Alphabet()
    def features_noalias(s,a):
        f = np.zeros(D)
        f[alphabet['action-bias',a]] = 1
        f[alphabet['state-bias',s]] = 1
        f[alphabet[s,a]] = 1      # no feature aliasing
        return f

    def features_aliased(s,a):
        f = np.zeros(D)
        # map so to it's aliased state.
        if Si.lookup(s) == (1,0): s = Si[0,1]
        if Si.lookup(s) == (0,0): s = Si[1,1]
        f[alphabet['action-bias',a]] = 1
        f[alphabet['state-bias',s]] = 1
        f[alphabet[s,a]] = 1
        return f

    # Student setup
    sfeatures = features_aliased
    sp = SoftmaxPolicy(M.A, D, sfeatures)

    # Student Policy
    student = sp.to_table(M)

    # Test cases
    test_lols(M, student)
    test_aggrevate(M, student)

def test_lols(M, student):
    """
    Test that we recover the desired limit cases for smooth lookahead teacher.
    """

    # LOLS is equivalent to one step of lookahead on the student (i.e., one call
    # to the Bellman operator, given on the student's value function).
    _, lols = M.B(M.V(student))
    teacher = smooth_lookahead_teacher(M, student, 0)[1]
    assert np.allclose(M.Advantage(lols),
                       M.Advantage(teacher))
    print("Passed Test LOLS")

def test_aggrevate(M, student):
    # the AggreVate teacher is equal to the optimal policy
    aggrevate = M.solve_by_value_iteration()['policy']
    teacher = smooth_lookahead_teacher(M, student, 1)[1]
    assert np.allclose(M.Advantage(aggrevate),
                       M.Advantage(teacher))
    print("Passed Test Aggrevate")

if __name__ == '__main__':
    main()
