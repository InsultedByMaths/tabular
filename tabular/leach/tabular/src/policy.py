import numpy as np
from scipy.optimize import minimize
from arsenal.maths.checkgrad import fdcheck
from arsenal.maths import softmax, onehot, sample
from arsenal.maths.stepsize import adam


class LinearPolicy(object):

    def __init__(self, A, D, features):
        self.A = A
        self.D = D
        self.weights = np.zeros(D)
        self.features = features
        self.update = adam(self.weights)

    def __call__(self, s):
        return sample(self.p(s))

    def to_table(self, M, F=None):
        if F is None: F = self.F(M)
        scores = F @ self.weights
        a = scores.argmax(axis=1)
        pi = np.zeros((M.S, M.A))
        pi[range(M.S), a] = 1
        return pi

    def F(self, M):
        F = np.zeros((M.S, M.A, self.D))
        for s in range(M.S):
            for a in range(M.A):
                F[s,a,:] = self.features(s, a)
        return F

    def scores(self, s):
        return np.array([self.weights.dot(self.features(s,a)) for a in range(self.A)])

    def solve_csc(self, M, rollin, F, rollout):
        """
        Solve the cost-sensitive classification method induced by (rollin,rollout).
        where `rollin` weights each state's importance and `rollout` assess
        the reward of each action in context.
        """
        return minimize(lambda w: self.csc_objective(w, M, rollin, F, rollout),
                        self.weights, jac=True).x

    def csc_objective(self, w, M, rollin, F, rollout):
        raise NotImplementedError


class NoRegretImitationLearning:

    def __init__(self, M):
        self.rollout = np.zeros((M.S, M.A))
        self.rollin = np.zeros(M.S)
        self.best = (-np.inf, None)
        self.i = 0

    def imitate(self,
                M,
                learner,
                Q: 'state-action rewards',
                F = None,
                ):

        self.i += 1

        # Perform dataset aggregation step.  Implemented below by keeping a
        # running sum of state-importance weights `rollin`, which are the
        # nonstationary part of dagger (i.e., the online learning part).
        self.rollin += M.d(learner.to_table(M, F))
        self.rollout += Q

#        if 0:
#            d = self.rollin / self.i
#            Q = self.rollout / self.i
#            w = learner.weights
#            _, grad = learner.csc_objective(w, M, d, F, Q, σ = 0.0001)
#            fdcheck(lambda: learner.csc_objective(w, M, d, F, Q, σ = 0.0001)[0],
#                    w, grad).show()

        learner.weights = learner.solve_csc(M,
                                            self.rollin / self.i,
                                            F,
                                            self.rollout / self.i)

        # track the running max over each iteration because we need to return
        # the best policy after all iterations, not the last one (an equally
        # weighted ensemble over iterations is also fine, but less good
        # theoretically, at least in the infinite sample case).
        J = M.J(learner.to_table(M, F))
        if J > self.best[0]:
            self.best = (J, learner.weights.copy())

        # set policy equal the best seen over iterations
#        self.weights = self.best[1]


class SoftmaxPolicy(LinearPolicy):

    def p(self, s):
        "conditional probability of each action in state s."
        return softmax(self.scores(s))

    def to_table(self, M, F=None):
        if F is None: F = self.F(M)
        return softmax(F @ self.weights, axis=1)

    def dlogp(self, s, a):
        "Compute ∇ log p(a | s)"
        d = np.zeros(self.D)
        d = self.features(s,a)
        p = self.p(s)
        for ap in range(self.A):
            d -= p[ap] * self.features(s, ap)
        return d

    def policy_gradient(self, M, A=None):
        "Policy gradient"
        # Compute the current policy as a table.
        pi = self.to_table(M)
        # Compute the performance (expected discounted return) under the current policy.
        J = M.J(pi)
        # Compute the discounted state distribution.
        d = M.d(pi)
        if A is None:
            A = M.Q(pi)
        # Initialize the gradient.
        g = np.zeros_like(self.weights)
        # Loop over all states and actions.
        for s in range(M.S):
            for a in range(M.A):
                # Accumulate the gradient term:
                # d(s) * pi(s,a) * (Q(s,a) - V(s)) * ∇ log π(a|s)
                g += d[s] * pi[s, a] * A[s, a] * self.dlogp(s, a)
        return J, g

    def natural_policy_gradient(self, M, A=None, damping=1e-3):
        """
        Compute the natural policy gradient.
        
        This method computes:
          - The usual policy gradient g = Σₛ d(s) Σₐ π(s,a) (Q(s,a) - V(s)) ∇ log π(s,a)
          - The Fisher information matrix F = Σₛ d(s) Σₐ π(s,a) [∇ log π(s,a)][∇ log π(s,a)]^T
          - Returns the natural gradient update: nat_grad = F⁻¹ g
          
        Parameters:
            M       : The MDP object.
            A       : (Optional) A precomputed Q-value array. If None, M.Q(pi) is used.
            damping : A small constant added to the diagonal of F for numerical stability.
            
        Returns:
            J         : The current performance (expected return) under the policy.
            nat_grad  : The natural gradient direction.
        """
        # Compute the current policy table.
        pi = self.to_table(M)
        # d: discounted state distribution (vector of length M.S)
        d = M.d(pi)
        # Compute Q-values; if not provided, use M.Q(pi)
        Q_vals = M.Q(pi) if A is None else A
        # Compute the baseline: V(s) = sum_a pi(s,a) * Q(s,a)
        V_vals = (pi * Q_vals).sum(axis=1)
        
        # Initialize the vanilla gradient and Fisher matrix.
        g = np.zeros_like(self.weights)
        F = np.zeros((self.D, self.D))
        
        # Loop over states and actions.
        for s in range(M.S):
            for a in range(M.A):
                # Advantage: Q(s,a) - V(s)
                advantage = Q_vals[s, a] - V_vals[s]
                # ∇ log π(a|s)
                grad_log = self.dlogp(s, a)
                # Accumulate the gradient (weighted by the state distribution and action probability)
                g += d[s] * pi[s, a] * advantage * grad_log
                # Accumulate the Fisher information matrix
                F += d[s] * pi[s, a] * np.outer(grad_log, grad_log)
        
        # Add damping to the Fisher matrix for numerical stability.
        F += damping * np.eye(self.D)
        
        # Solve for the natural gradient: F * nat_grad = g
        nat_grad = np.linalg.solve(F, g)
        
        # Compute the current performance J for reporting.
        J = M.J(pi)
        return J, nat_grad

    def csc_objective(self, w, M, d, F, Q, σ = 0.0001):
        # Warning: doesn't do aggregate because we I don't have a good reduction
        # for the costs in logistic regression yet.

        ww = self.weights   # use weights passed in and restore after
        self.weights = w

        # This is really just logistic regression with example weights `d` and
        # labels equal to argmax `expert_Q`.

        scores = F.dot(w)
        pi = softmax(scores, axis=1)
        y = Q.argmax(axis=1)
        J = d @ np.log(pi[range(M.S), y])
        g = d @ F[range(M.S), y]
        g -= np.einsum('sa,sad,s->d', pi, F, d)

        # L2 regularization to keep the objective strongly convex and avoid
        # pathologies, which occur in logistic regression when there are
        # features that only occur with one class and get infinite weight.
        J -= σ * 0.5 * self.weights.dot(self.weights)
        g -= σ * self.weights

        self.weights = ww   # restore weights

        return -J, -g


class LinearArgmaxPolicy(LinearPolicy):

    def p(self, s):
        "conditional probability of each action in state s."
        return onehot(self.scores(s).argmax(), self.A)

    # TODO: we can solve this optimization faster with a wieghted least-squares solver.
    def csc_objective(self, w, M, d, F, Q):
        # Reduction from CSC to linear regression
        ww = self.weights   # use weights passed in and restore after
        self.weights = w
        diff = (F.dot(w) - Q)
        J = np.einsum('s,sa->', d, np.square(diff))
        g = np.einsum('s,sa,sad->d', d, diff, F)
        self.weights = ww   # restore weights
        return 0.5 * J, g

    def solve_csc(self, M, rollin, F, rollout):
        #from arsenal.maths import wls

        FF = np.zeros((M.S*M.A, self.D))
        ri = np.zeros(M.S*M.A)
        ro = np.zeros(M.S*M.A)
        for s in range(M.S):
            for a in range(M.A):
                FF[s*M.A + a, :] = F[s,a,:]
                ri[s*M.A + a] = rollin[s]
                ro[s*M.A + a] = rollout[s,a]

        #zz = wls(FF, ro, ri)

        ww = minimize(lambda w: self.csc_objective(w, M, rollin, F, rollout), self.weights, jac=True).x
        #from arsenal.maths import compare
        #compare(ww, zz).show()

        #return zz
        return ww
