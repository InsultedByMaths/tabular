# -*- coding: utf-8 -*-
import random
import numpy as np
import itertools
import collections
from collections import defaultdict
import copy
import pdb, traceback, sys
import itertools
import random
import pdb

from leach.tabular.src.mdp import DiscountedMDP
from leach.tabular.src.gridworld import GridWorld

from arsenal import iterview, Alphabet

# Last dimension is the distribution.
dirichlet = lambda *size: np.random.dirichlet(np.ones(size[-1]), size=size[:-1])

class FixToyMDP(object):
    def __init__(self):
        self.Si = None
        self.G = None
        self.M = None

    def convert(self, Vest):
        return {self.Si.lookup(s): Vest[s] for s in range(self.M.S)}


    def __call__(self, mapname="3x6.txt", gamma=0.99):
        G = GridWorld(mapname=mapname)
        [values, Si, _] = G.encode() # [MDP, State-lookups, Action-lookups]
        M = DiscountedMDP(*values, gamma=gamma)
        D = M.S * M.A + M.S + M.A

        self.Si = Si
        self.G = G
        self.M = M

        alphabet = Alphabet()

        def features_noalias(s,a):
           f = np.zeros(D)
           f[alphabet['action-bias',a]] = 1
           f[alphabet['state-bias',s]] = 1
           f[alphabet[s,a]] = 1      # no feature aliasing
           return f

        #def features(s,a):
        #    f = np.zeros(D)
        #    # map so to it's aliased state.
        #    if Si.lookup(s) == (1,0): s = Si[0,1]
        #    if Si.lookup(s) == (0,0): s = Si[1,1]
        #    f[alphabet['action-bias',a]] = 1
        #    f[alphabet['state-bias',s]] = 1
        #    f[alphabet[s,a]] = 1
        #    return f

        
        # def features(s,a):
        #     pdb.set_trace()
        #     f = np.zeros(D)
        #     # map so to it's aliased state.
        #     # (0,0) -> (2,0)
        #     # (0,2) -> (2,2)
        #     if Si.lookup(s) == (0,1) or Si.lookup(s) == (0,2) or \
        #        Si.lookup(s) == (1,1) or Si.lookup(s) == (1,2): s = Si[2,0]

        #     if Si.lookup(s) == (0,0) or Si.lookup(s) == (2,1): s = Si[2,2]
        #     f[alphabet['action-bias',a]] = 1
        #     f[alphabet['state-bias',s]] = 1
        #     f[alphabet[s,a]] = 1
        #     return f

        return M, features_noalias, D

class GarnetMDP(object):

    def _mdp(self, S, A, gamma, b=None, r=None):
        """Randomly generated MDP

        Text taken from http://www.jmlr.org/papers/volume15/geist14a/geist14a.pdf

        More precisely, we consider Garnet problems (Archibald et al., 1995), which are a class
        of randomly constructed finite MDPs. They do not correspond to any specific application,
        but are totally abstract while remaining representative of the kind of MDP that might be
        encountered in practice. In our experiments, a Garnet is parameterized by 3 parameters
        and is written G(S, A, b): S is the number of states, A is the number of actions, b
        is a branching factor specifying how many possible next states are possible for each state-action
        pair (b states are chosen uniformly at random and transition probabilities are set
        by sampling uniform random b − 1 cut points between 0 and 1). The reward is state-dependent:
        for a given randomly generated Garnet problem, the reward for each state is uniformly sampled
        between 0 and 1.

        The discount factor γ is set to 0.95 in all experiment

        We consider two types of problems, “small” and “big”, respectively
        corresponding to instances G(30, 2, p=2, dim=8) and G(100, 4, p=3, dim=20)

        """

        if b is None: b = S
        if r is None: r = S

        P = np.zeros((S,A,S))
        states = np.array(list(range(S)))

        #rs = np.random.choice(states, size=r, replace=False)

        for s in range(S):
            for a in range(A):
                # pick b states to be connected to.
                connected = np.random.choice(states, size=b, replace=False)
                P[s,a,connected] = dirichlet(b)

        # how many states get rewards
        #R[:,:,s] = np.random.uniform(0,1,size=(S,A,S))
        R = np.zeros((S,A,S))
        R[:,:,S-1] = 1

        return DiscountedMDP(
            s0 = dirichlet(S),
            R = R,
            P = P,
            gamma = gamma,
        )

    def __call__(self, S=None, A=None, gamma=None, b=None, H=None):
        M = self._mdp(S, A, gamma, b=b)

        h = np.random.randint(0,H,size=S).astype(int)

        D = S*A
        alphabet = Alphabet()
        def features(s,a):
            s = h[s]
            f = np.zeros(D)
            f[alphabet[s,a]] = 1
            return f

        return M, features, D
