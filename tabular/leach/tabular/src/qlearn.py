import numpy as np

# TODO: add tests for Q learning, SARSA, and expected-SARSA.
class QLearner:

    def __init__(self, M, Si=None):
        [self.S, self.A, _] = M.P.shape
        self.Q = np.zeros((M.S, M.A))
        #self.Q = np.random.rand(M.S, M.A)
        self.epsilon = 1.0
        self.gamma = M.gamma
        self.t = 0
        self.alpha = 1.0
        self.Si = Si

    def features(self, s):
        # TODO: Fix Hack
        if self.Si.lookup(s) == (0,1) or self.Si.lookup(s) == (0,2) or \
           self.Si.lookup(s) == (1,1) or self.Si.lookup(s) == (1,2):
            s = self.Si[2,0]

        if self.Si.lookup(s) == (0,0) or self.Si.lookup(s) == (2,1):
            s = self.Si[2,2]
        return s

    def __call__(self, s):
        if np.random.uniform() <= self.epsilon:
            return int(np.random.randint(self.A))
        if self.Si is not None:
            s = self.features(s)
        return np.argmax(self.Q[s,:])

    def to_table(self):
        pi = np.zeros((self.S, self.A))
        for s in range(self.S):
            pi[s, :] = self.p(s)
        return pi

    def p(self, s):
        if self.Si is not None:
            s = self.features(s)

        pi = np.zeros(self.A)
        pi[np.argmax(self.Q[s,:])] = 1-self.epsilon
        pi[:] += self.epsilon/self.A
        return pi

    def update(self, s, a, r, sp):
        if self.Si is not None:
            s = self.features(s)

        self.t += 1
        self.Q[s,a] = ((1-self.alpha)*self.Q[s,a] +
                       self.alpha*(r + self.gamma*max(self.Q[sp,:])))
