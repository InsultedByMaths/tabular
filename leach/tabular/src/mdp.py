import numpy as np
import warnings
from arsenal.maths import sample #, wls
from numpy.random import uniform
from scipy import linalg
import pdb


# Last dimension is the distribution.
dirichlet = lambda *size: np.random.dirichlet(np.ones(size[-1]), size=size[:-1])


def random_MDP(S, A, gamma, b=None, r=None):
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

    ## Compute Optimal plicy
    #M = DiscountedMDP( s0 = dirichlet(S), R = R, P = P, gamma = gamma)
    #opt = M.solve()['policy']

    ## Perturbing the transition matrix
    #move = 20
    #pert = 5

    ##P = np.zeros((S,A,S))
    #P = np.copy(P)
    #gamma_t = np.random.random_sample((S,))
    #for s in range(S):
    #    for a in range(A):
    #        opt_a = np.argmax(opt[s])
    #        sp = np.argmax(M.P[s,a])

    #        if a != opt_a:
    #            for sp in range(S):
    #                theta = np.random.randint(-move, move)
    #                idx = (theta+sp) % S
    #                P[s,a,idx] *= (1 + gamma_t[idx] * pert)
    #from numpy.linalg import norm
    #linfnorm = norm(P, axis=2, ord=1)
    #M.P = P.astype(np.float) / linfnorm[:,:,None]
    #return M


class MDP(object):
    def __init__(self, s0, P, R):
        # P: Probability distribution p(S' | A S) stored as an array S x A x S'
        # R: Reward function r(S, A, S) -> Reals stored as an array S x A x S'
        # s0: Distribution over the initial state.
        self.s0 = s0
        [self.S, self.A, _] = P.shape
        self.P = P
        self.R = R

    def r_sa(self):
        "Compute `r(s,a)` from `r(s,a,s')`"
        return np.einsum('sap,sap->sa', self.P, self.R)


class FiniteHorizonMDP(MDP):
    "Finite-horizon MDP."

    def __init__(self, s0, P, R, T):
        super(FiniteHorizonMDP, self).__init__(s0, P, R)
        self.T = T

    def value(self, policy):
        "Compute `T`-step value functions for `policy`."
        S = self.S; A = self.A; P = self.P; R = self.R; T = self.T
        Q = np.zeros((T+1,S,A))
        V = np.zeros((T+1,S))
        for t in reversed(range(T)):
            for s in range(S):
                for a in range(A):
                    Q[t,s,a] = P[s,a,:] @ (R[s,a,:] + V[t+1,:])
                V[t,s] = policy[s,:] @ Q[t,s,:]
        J = self.s0 @ V[0,:]
        # Value functions: No conditioning, state conditioned, state-action conditioned
        return J, V, Q

    def d(self, policy):
        "Probability of state `s` under the given `policy` conditioned on each time `t <= T`."
        S = self.S; A = self.A; P = self.P; T = self.T
        d = np.zeros((T, S))
        d[0,:] = self.s0
        for t in range(1, T):
            for sp in range(S):
                d[t,sp] = sum(d[t-1,s] * policy[s,a] * P[s,a,sp] for s in range(S) for a in range(A))
            d[t] /= d[t].sum()   # normalized per-time step.
        return d


class MarkovChain(object):
    "γ-discounted Markov chain."
    def __init__(self, s0, P, gamma):
        self.s0 = s0
        self.P = P
        self.gamma = gamma

    def start(self):
        return sample(self.s0)

    def step(self, s):
        if uniform(0,1) <= 1-self.gamma:
            return self.start()
        return sample(self.P[s,:])

    def P_with_reset(self):
        "Transition matrix with (1-γ)-reset dynamics."
        return (1-self.gamma)*self.s0[None,:] + self.gamma*self.P

    def d(self):
        "Stationary distribution."
        # The stationary distribution, much like other key quantities in MRPs,
        # is the solution to a linear recurrence,
        #            d = (1-γ) s0 + γ Pᵀ d    # transpose because P is s->s' and we want s'->s.
        #   d - γ Pᵀ d = (1-γ) s0
        # (I - γ Pᵀ) d = (1-γ) s0
        # See also: stationarity condition in the linear programming solution
        return linalg.solve(np.eye(self.S) - self.gamma * self.P.T,  # note the transpose
                            (1-self.gamma) * self.s0)

    def d_via_eigen(self):
        """
        Compute the stationary distribution via eigen methods.
        """
        # Markov chain has this transition matrix, which makes the transition to
        # the start state explicit.
        t = self.P_with_reset()

        # Transition matrix is from->to, so it sum-to-one over rows so we
        # transpose it.  Alternatively, we can get do the left eig decomp.
        [S, U] = linalg.eig(t.T)

        ss = np.argsort(S)
        S = S[ss]
        U = U[:,ss]

        s = U[:,-1].real
        s /= s.sum()
        return s

    def successor_representation(self):
        "Dayan's successor representation."
        return linalg.solve(np.eye(self.S) - self.gamma * self.P,
                            np.eye(self.S))

    def eigenvalue_stationary(self):
        "Stationary distribution Eigen Values"
        pi = np.random.rand(13,1)
        for _ in range(100000): pi = pi.T.dot(self.P)
        return pi


class MRP(MarkovChain):
    "Markov reward process."
    def __init__(self, s0, P, R, gamma):
        super(MRP, self).__init__(s0, P, gamma)
        self.R = R
        self.gamma = gamma
        [self.S, _] = P.shape
        assert R.ndim == 1 and R.shape[0] == P.shape[0] == P.shape[1]

    def V(self):
        "Value function"
        return linalg.solve(np.eye(self.S) - self.gamma * self.P, self.R)

    def J(self):
        "Expected value of MRP"
        # Expected cumulative (discounted) reward achieved by following the current policy.
        #return self.V() @ self.s0
        return self.R @ self.d() / (1-self.gamma)

    def run(self):
        "Simulate the MRP"
        s = self.start()
        while True:
            sp = self.step(s)
            yield s, self.R[s], sp
            s = sp

    def P_with_reset(self):
        "Transition matrix with (1-γ)-reset dynamics."
        return (1-self.gamma)*self.s0[None,:] + self.gamma*self.P

    #def vfa_exact(self, F):
    #    "Solve least-squared regression to the true value function."
    #    # free to used other weighting schemes
    #    return wls(F, self.V(), self.d())

    def projected_bellman_error(self, F, v, w):
        """Average projected Bellman error under the MRP's stationary distribution.
        This quantity will go to zero in all solution methods, which minimize it.
        """
        w = np.diag(w)
        proj = F @ linalg.inv(F.T @ w @ F) @ F.T @ w
        resid = proj @ self.bellman_residual(v)
        return resid @ resid

    def vfa_fit(self, F, weights=None):
        "Projected Bellman Residual Minimization."

        # weighted projection onto span(F)
        [S, D] = F.shape
        assert S == self.S

        w = np.diag(weights) if weights is not None else np.eye(F.shape[0])

        # To work in the larger space use the following projection operator
        #Π = (F @ linalg.inv(F.T @ w @ F) @ F.T @ w).dot
        #  = F @ linalg.pinv(F)
        #A = np.eye(self.S) - self.gamma*Π(self.P)
        #b = Π(self.R).T
        #return np.linalg.solve(A, b)

        A = F.T @ w @ (F - self.gamma*self.P @ F)
        b = F.T @ w @ self.R

        assert A.shape == (D, D) and b.shape == (D,)
        sol = np.linalg.solve(A, b)
        assert sol.shape == (D,)
        return sol

    # TODO: untested
    def vfa_lp(self, F, mu, verbose=False):
        # TODO: add L1 regularization like in the MDP method.
        gamma = self.gamma; R = self.R; P = self.P; S = self.S
        from gurobipy import Model, GRB
        m = Model(); m.modelSense = GRB.MINIMIZE
        [_,D] = F.shape; mu = np.ones(S)/S if mu is None else mu
        w = [m.addVar(lb=-GRB.INFINITY) for _ in range(D)]
        v = [m.addVar(lb=-GRB.INFINITY, obj=(1-gamma)*mu[s]) for s in range(S)]
        for s in range(S):
            m.addConstr(v[s] == sum(F[s,k] * w[k] for k in range(D)))
            m.addConstr(v[s] >= R[s] + gamma*sum(P[s,sp] * v[sp] for sp in range(S)))
        if not verbose: m.setParam('OutputFlag', False)   # quiet mode.
        m.optimize()
        if m.Status != 2: raise ValueError(m)
        return (np.array([v[s].x for s in range(S)]),
                np.array([w[k].x for k in range(D)]))

    def vfa_indirect(self, F):
        """
        Solve for value function parameters indirectly by linearly approximating `R`
        and `P` in feature space `F`. The value function in can be solved for in
        that space.  The benefit of this is that the `F` space will be
        D-dimensional where D << |S|, which makes solving the linear system much
        cheaper, O(D³) vs. O(|S|³).  This is the basis of LSTD methods.

        # see also: `vfa_direct`
        """
        # Effectively, we compile the MRP into a "linear MDP" with an equivalent solution
        D = F.shape[1]
        r = linalg.lstsq(F, self.R)[0]        # find r that approximates reward function
        p = linalg.lstsq(F, self.P @ F)[0]    # find p that best-approximates features of next state.
        return linalg.solve(np.eye(D) - self.gamma*p, r)  # solve for value function in the feature space

    def vfa_td(self, F, λ, steps, α = 10, avg = False, verbose = False):
        "TD(λ) learning with features."
        γ = self.gamma; D = F.shape[1]
        θ = np.zeros(D); w = np.zeros(D)
        z = np.zeros(D)
        t = 0
        for t, (s, r, sp) in enumerate(self.run(), start=1):
            if t == steps: break
            #δ = r + γ * (F[sp] @ θ) - F[s] @ θ
            δ = F[s] @ θ - (r + γ * (F[sp] @ θ))
            z = F[s] + γ * λ * z
            θ -= α/np.sqrt(t) * δ * z
            if avg: w += θ   # iterate averaging seems to help a LOT here.
            if verbose and t % 5000 == 0:
                from arsenal.viz import lc
                v = (w/t) if avg else θ
                Δ = linalg.norm(self.bellman_residual(F @ v))
                print(t, Δ, np.log(Δ))
                lc['td'].yscale = 'log'
                lc['td'].update(t, err = Δ)
        return w/t if avg else θ

    def vfa_lstd(self, F, λ, steps, callback = None, verbose = False):
        "LSTD learning with features."
        # TODO: add Boyan's LSTD(λ) extension. Beware of Csaba's cautions in the
        # LSTD section.  Also need to understand the RLSTD ideas. There is also
        # least-squares policy evalation.
        γ = self.gamma; D = F.shape[1]
        A = np.eye(D)
        b = np.zeros(D)
        t = 0
        for t, (s, r, sp) in enumerate(self.run(), start=1):
            if t == steps: break
            A += np.outer(F[s], F[s] - γ * F[sp])
            b += r * F[s]
            if verbose and t % 5000 == 0:
                θ = linalg.solve(A/t, b/t)
                callback(t, θ)
        return linalg.solve(A/t, b/t)

    def bellman_residual(self, v):
        return (self.R + self.gamma*self.P @ v) - v

    def epsilon_return_mixing_time(self):
        """The epsilon-return mixing time is the smallest truncated
        value function with a epsilon bounded estimation at all state (i.e.,
        under infinity norm).

        The quantity is upper bounded by

        H(epsilon) <= log_\gamma( epsilon * (1-gamma) / Rmax )

        """
        t = 0
        V = self.V()            # true value function
        Vt = np.zero(self.S)    # truncated-time estimate of value function
        while True:
            t += 1
            [Vt, _] = self.B(V)  # how many times do we have to apply the Bellman operator
            err = np.abs(V - Vt).max()   # infinity-norm
            yield t, err, Vt


class DiscountedMDP(MDP):
    "γ-discounted, infinite-horizon Markov decision process."
    def __init__(self, s0, P, R, gamma):
        # γ: Temporal discount factor
        super(DiscountedMDP, self).__init__(s0, P, R)
        self.gamma = gamma

    def copy(self):
        return DiscountedMDP(self.s0.copy(),
                             self.P.copy(),
                             self.R.copy(),
                             self.gamma * 1)

    def run(self, learner):
        s = self.start()
        while True:
            a = learner(s)
            r, sp = self.step(s, a)
            if not learner.update(s, a, r, sp):
                break
            s = sp

    def step(self, s, a):
        sp = sample(self.P[s,a,:])
        r = self.R[s,a,sp]
        # TODO: should be equivalent (at least in some sense) to a version which
        # runs "episodes" for an expected 1/(1-gamma) steps.  (this version is
        # less stateless than this one because there isn't a variable which
        # controls the reset that persists across calls.  The current version
        # also doesn't assume the caller calls it in sequence, which is useful
        # for algorithms that assume unrestricted access to a so called
        # "generative model" of the environment) In order to exactly simulate
        # the setup below, we need to consider other properties of the
        # distribution, i.e., variance and higher order moments.  I think it can
        # all be worked out for this simple case because the random reset in
        # independent.  This is closely related to Ben Libdit.  Also, should be
        # equivalent to transitions with P_with_reset.
        if uniform(0,1) <= 1-self.gamma:
            sp = self.start()
        return r, sp

    def start(self):
        return sample(self.s0)

    def P_with_reset(self):
        "Transition matrix with (1-γ)-reset dynamics."
        # TODO: we might need to be careful when using this to make sure that
        # the reward is given before the state is restarted.
        return (1-self.gamma)*self.s0[None,None,:] + self.gamma*self.P

    def mrp(self, policy):
        "MDP becomes an `MRP` when we condition on `policy`."
        return MRP(self.s0,
                   np.einsum('sa,sap->sp', policy, self.P),
                   np.einsum('sa,sap,sap->s', policy, self.P, self.R),
                   self.gamma)

    def J(self, policy):
        "Expected value of `policy`."
        # Expected cumulative (discounted) reward achieved by following the current policy.
        return self.mrp(policy).J()

    def V(self, policy):
        "Value function for `policy`."
        return self.mrp(policy).V()

    def successor_representation(self, policy):
        "Dayan's successor representation."
        return self.mrp(policy).successor_representation()

    def d(self, policy):
        "γ-discounted stationary distribution over states conditioned `policy`."
        return self.mrp(policy).d()

    def Q(self, policy):
        "Compute the action-value function `Q(s,a)` for a policy."
        # See also: Q_by_linalg
        v = self.mrp(policy).V()
        r = self.r_sa()
        Q = np.zeros((self.S, self.A))

        for s in range(self.S):
            for a in range(self.A):
                Q[s,a] = r[s,a] + self.gamma*self.P[s,a,:] @ v
        return Q

    def Q_by_linalg(self, π):
        """
        Compute the action-value function `Q(s,a)` for a policy `π` by solving
        a linear system of equations.
        """
        # Notes: the implementation of the method is kind of messy (because of
        # all the reshaping business).  Also, the linear system for `V` is much
        # cleaner and more efficient because the linear system is smaller by a
        # factor of A.  So I've made computing Q by V the defualt method.
        Π = np.zeros((self.S, self.S*self.A))
        for s in range(self.S):
            for a in range(self.A):
                Π[s, self.A*s + a] = π[s,a]
        P = self.P.reshape((self.S*self.A, self.S))
        r = self.r_sa().ravel()
        return (linalg.solve(np.eye(self.S*self.A) - self.gamma * P @ Π, r)
                .reshape((self.S, self.A)))

    def Advantage(self, policy):
        "Advantage function for policy."
        return self.bellman_residual(self.V(policy))

    def B(self, V):
        "Bellman operator."
        # Act greedily according to one-step lookahead on V.
        Q = self.Q_from_V(V)
        pi = np.zeros((self.S, self.A))
        pi[range(self.S), Q.argmax(axis=1)] = 1
        v = Q.max(axis=1)
        return v, pi

    def Q_from_V(self, V):
        "Lookahead by a single action from value function estimate `V`."
        return (self.P * (self.R + self.gamma * V[None,None,:])).sum(axis=2)

    def bellman_residual(self, V):
        "The control case of the Bellman residual."
        return self.Q_from_V(V) - V[:,None]

    def apply_potential_based_shaping(self, phi):
        "Apply potential-based reward shaping"
        self.R[...] = self.shaped_reward(phi)

    def shaped_reward(self, phi):
        # Potential based shaping augments the reward function with an extra
        # term:  R'(s,a,s') = R(s,a,s') + γ ϕ(s') - ϕ(s)
        # See also: performance-difference lemma
        γ = self.gamma
        #r = np.zeros_like(self.R)
        #for s in range(self.S):
        #    for a in range(self.A):
        #        for sp in range(self.S):
        #            r[s,a,sp] = self.R[s,a,sp] + γ*phi[sp] - phi[s]
        r = self.R + γ*phi[None,None,:] - phi[:,None,None]
        return r

    def apply_look_adhead_shaping(self, phi):
        """ Principled Methods for Advising Reinforcement Learning Agents
            https://pdfs.semanticscholar.org/b8b6/b33f750b93b3fb0b90ef219f7e0d0acc66aa.pdf
        """
        #self.R[...] = self.shaped_look_adhead_reward(phi)
        r = np.zeros_like(self.R)
        γ = self.gamma
        values = [(s,a,sp,ap)
                   for s in range(self.S)
                   for a in range(self.A)
                   for sp in range(self.S)
                   for ap in range(self.A)]

        for (s,a,sp,ap) in values:
            r[s,a,sp] += self.R[s,a,sp] + γ*phi[sp, ap] - phi[s,a]

        self.R = r

    def apply_look_behind_shaping(self, phi):
        """ Principled Methods for Advising Reinforcement Learning Agents
            https://pdfs.semanticscholar.org/b8b6/b33f750b93b3fb0b90ef219f7e0d0acc66aa.pdf
        """
        r = np.zeros_like(self.R)
        γ = self.gamma

        values = [(s,a,sp,ap)
                  for s in range(self.S)
                  for a in range(self.A)
                  for sp in range(self.S)
                  for ap in range(self.A)]

        for (s,a,sp,ap) in values:
            r[s,a,sp] += self.R[s,a,sp] + phi[s, a] - (1/γ) *phi[sp,ap]

        self.R = r


    def shaped_look_adhead_reward(self, phi):
        # Potential based shaping augments the reward function with an extra
        # term:  R'(s,a,s') = R(s,a,s') + γ ϕ(s') - ϕ(s)
        γ = self.gamma
        r = self.R + γ*phi[None,None,:] - phi[:,None,None]
        return r


#    def apply_disadvantage_based_shaping(self, phi):
#        "Apply potential-based reward shaping"
#        # Potential based shaping augments the reward function with an extra
#        # term:  R'(s,a,s') = R(s,a,s') + γ ϕ(s') - ϕ(s)
#        # See also: performance-difference lemma
#        S = self.S; A = self.A
#        for s in range(S):
#            for a in range(A):
#                for sp in range(S):
#                    self.R[s,a,sp] = phi[s] - (self.R[s,a,sp] + self.gamma*phi[sp])

#    def apply_thor_shaping(self, phi, policy, k=2):
#        "Apply potential-based reward shaping"
#        # term:  R'(s,a,s') = Σ^{k} R(s,a,s') + γ ϕ(s') - ϕ(s)
#        S = self.S; A = self.A
#        for s in range(S):
#            s1 = si = s
#            for a in range(A):
#               for sp in range(S):
#                    reward = []
#                    sk = sp
#                    for i in range(k):
#                       reward.append((self.gamma**i) * self.R[si,a,sk])
#                       si = sk
#                       a = policy(sk)
#                       sk = sample(M.P[s,a,:])
#                    reward = np.array(reward)
#                    reward += (self.gamma**steps) *  phi[sk] - phi[s1]
#                    self.R[sk,a,sp] = reward


    def solve_by_policy_iteration(self, max_iter=1000):
        "Solve the MDP with the policy iteration algorithm."
        V = np.zeros(self.S)
        pi = pi_prev = np.zeros((self.S, self.A))
        #while True:
        for _ in range(max_iter):
            # Policy iteration does not take the value function from Bellman
            # operator (the variable `_` below). Instead, it uses the greedy
            # policy. (the greedy value function doesn't satisfy the first
            # Bellman equation). We find a new value function for the improved
            # policy, which satisfies the first Bellman equation.
            _, pi = self.B(V)  # Bellman equation 2: the optimal policy is greedy with its value function
            V = self.V(pi) # Bellman equation 1: definition of value function for a policy
                               # Note: the value function returned by `B` is not the same as policy evaluation
            if (pi_prev == pi).all(): break
            pi_prev = pi

        if (pi_prev != pi).all():
            warnings.warn('policy iteration exceeded max iterations')

        return {
            'obj': V @ self.s0,
            'policy': pi,
            'V': V,
        }

    def solve_by_value_iteration(self, tol=1e-10, max_iter=1000):
        "Solve the MDP with the value iteration algorithm."
        V = np.zeros(self.S)
        for _ in range(max_iter):
            V1, pi = self.B(V)
            if np.abs(V1 - V).max() <= tol: break
            V = V1
        if np.abs(V1 - V).max() > tol:
            warnings.warn('value iteration exceeded max iterations')
        # Bounding the difference ||V_t - V_{t-1}||_inf < tol
        # bounds ||V_{greedy policy wrt V_t} - V*||_inf < 2*tol*\gamma/(1-\gamma),
        # which is a more meaningful bound.
        return {
            'obj': V @ self.s0,
            'policy': pi,
            'V': V,
        }

    def solve_by_lp(self):
        "Solve the MDP by primal linear programming."

        # References:
        # http://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/slides/mdps.pdf

        # The straightforward formulation of policy optimization is not al LP:
        #
        #    maximize \sum_{s,a} r(s,a) π(a ∣ s) ⋅ μ(s)
        #
        # because μ (the distribution over states) is multiplied by the policy π.
        #
        # Luckily, there is a simple trick to avoid this nonlinearity. What we
        # do is merge `π(a ∣ s)` and `μ(s)` into `μ(s,a) = π(a ∣ s) ⋅ μ(s)`.
        #
        #   linear objective:   maximize ∑_{s,a} r(s,a) μ(s,a)
        #
        # can recover the original variables by marginalization,
        #
        #    μ(s) = ∑_a μ(s,a)  and  π(a ∣ s) = μ(s,a) / μ(s)
        #
        from gurobipy import Model, GRB
        P = self.P; S = list(range(self.S)); A = list(range(self.A)); γ = self.gamma
        m = Model('policy-opt')
        m.modelSense = GRB.MAXIMIZE

        # Compile away s' in (immediate) reward function.
        r = self.r_sa()

        # Define objective and constrain μ(s) ≥ 0.
        μ = {(s,a): m.addVar(lb=0, obj=r[s,a]) for s in S for a in A}

        # Stationarity of μ(s,a):
        #  - Notes: In this constraint, `s0` is arbitrary: all you /actually/ need
        #    there is *some* distribution over s', e.g., a uniform distribution is
        #    fine. Actually, any positive function is fine, but then you have to
        #    renormalize to interpret μ as a joint distribution.
        for sp in S:
            m.addConstr(sum(μ[sp, ap] for ap in A)
                        == (1-γ) * self.s0[sp] + γ * sum(μ[s,a] * P[s,a,sp] for s in S for a in A))

        # Sum-to-one constraints are not necessary. Furthermore, if we do these
        # constraints they (superficially) mess up the interpertation of the
        # dual variables as value functions. this constraint, it shifts the
        # value-function interpretation of the Lagrange multipliers.
        # Specifically, if this is the last constraint (i.e., in the -1
        # position), V = λ[:-1] + λ[-1]

        m.setParam('OutputFlag', False)   # quiet mode.
        m.optimize()
        assert m.Status == 2

        # Extract solution
        mm = np.zeros((self.S, self.A))
        pi = np.zeros((self.S, self.A))
        for s in S:
            for a in A:
                mm[s,a] = μ[s,a].x
        for s in S:
            if mm[s].sum() == 0:
                pi[s,:] = 1/self.A
            else:
                pi[s,:] = mm[s,:] / mm[s].sum()

        return {
            'obj': m.getObjective().getValue() / (1-γ),
            'policy': pi,
            'mu': mm.sum(axis=1),
            'mm': mm,
            'V': np.array([c.Pi for c in m.getConstrs()]),
            'model': m,
        }

    def solve_by_dual_lp(self):
        "Solve the MDP by dual linear programming."
        P = self.P; S = range(self.S); A = range(self.A); γ = self.gamma
        r = self.r_sa()

        from gurobipy import Model, GRB

        m = Model('dual')
        m.modelSense = GRB.MINIMIZE
        V = [m.addVar(obj=(1-γ)*self.s0[s],
                      lb=-GRB.INFINITY) for s in S]  # note: no lb because primal is equality constraint.

        for s in S:
            for a in A:
                m.addConstr(
                    V[s] >= r[s,a] + γ*sum(P[s,a,sp] * V[sp] for sp in S)
                )

        m.setParam('OutputFlag', False)   # quiet mode.
        m.optimize()
        assert m.Status == 2

        # Extract solution
        v = np.zeros(self.S)
        for s in S:
            v[s] = V[s].x

        pi = np.zeros((self.S, self.A))
        for s in S:
            q = r[s,:] + γ*P[s,:,:] @ v
            a = q.argmax()
            pi[s,a] = 1

        return {
            'obj': m.getObjective().getValue() / (1-γ),
            'policy': pi,
            'mu': np.array([c.Pi for c in m.getConstrs()]),
            'V': v,
            'model': m,
        }

    # default solution method is policy iteration.
    #solve = solve_by_policy_iteration
    #solve = solve_by_dual_lp
    #solve = solve_by_value_iteration
    solve = solve_by_policy_iteration
