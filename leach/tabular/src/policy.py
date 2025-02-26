import numpy as np
from scipy.optimize import minimize
from arsenal.maths.checkgrad import fdcheck
from arsenal.maths import softmax, onehot, sample
from arsenal.maths.stepsize import adam
import pdb


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

    def natural_policy_gradient(self, M, pi, A=None, damping=1e-3):
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
        # pi = self.to_table(M)
        # d: discounted state distribution (vector of length M.S)
        d = M.d(pi)
        # Get Q-values.
        if A is None:
            Q_vals = M.Q(pi)

        # V(s) = sum_a pi(s,a)*Q(s,a)
        V_vals = M.V(pi)
        
        # Initialize the vanilla gradient and Fisher matrix.
        g = np.zeros_like(self.weights)
        #F = np.zeros((self.D, self.D))
        F = np.zeros((M.S, M.S))
        
        # Loop over states and actions.
        for s in range(M.S):
            for a in range(M.A):
                f_forward = np.zeros((M.S, M.S))
                f_forward[s,s] = 1
                # ∇ log π(a|s)
                grad_log = self.dlogp(s, a)
                # Accumulate the gradient (weighted by the state distribution and action probability)
                g += d[s] * pi[s, a] * (Q_vals[s, a] - V_vals[s]) * grad_log
                # Accumulate the Fisher information matrix
                F += d[s] * pi[s, a] * np.outer(grad_log, grad_log)
                # F += d[s] * pi[s,a] * f_forward * (1/pi @ (1/pi).T)
        
        # Add damping to the Fisher matrix for numerical stability.
        # F += damping * np.eye(self.D)
        F += damping * np.eye(M.S)
        
        # Solve for the natural gradient: F * nat_grad = g
        nat_grad = np.linalg.solve(F, g)
        
        # Compute the current performance J for reporting.
        J = M.J(pi)
        return J, nat_grad

    def dual_mirror_descent_update(self, M, learning_rate, A=None):
        """
        Perform a dual update on the policy via online mirror descent (OMD)
        in the dual (logits) space. This implements the update

            log pi_{t+1}(a|s) = log pi_t(a|s) + eta * A(s,a) 

        for each state s and action a.

        Then, we update the underlying weight vector by solving a least-squares
        problem to find w such that the induced logits (i.e., F @ w) approximate the 
        new logits.
        
        Parameters:
            M             : The MDP object.
            learning_rate : The step size (eta) for the dual update.
            A             : (Optional) Q-value array. If None, uses M.Q(pi).
        
        Returns:
            J             : The current performance under the updated policy.
        """
        # Compute the current policy as a table and the state distribution.
        pi = self.to_table(M)
        d = M.d(pi)
        
        # Get Q-values.
        if A is None:
            Q_vals = M.Q(pi)

        # V(s) = sum_a pi(s,a)*Q(s,a)
        V_vals = M.V(pi)
        
        # Compute updated logits for each state-action pair.
        # new_logits(s,a) = log(pi(s,a)) + learning_rate * A(s,a)
        S, A_num = M.S, M.A
        new_logits = np.zeros((S, A_num))
        for s in range(S):
            for a in range(A_num):
                # Add a small constant to avoid log(0)
                new_logits[s, a] = np.log(pi[s, a] + 1e-8) + d[s] * pi[s, a] * learning_rate * (Q_vals[s, a] - V_vals[s])
        
        # Now, update the underlying weight vector.
        # We have the relationship:
        #    logits(s,a) = features(s,a)^T * weights
        # Let F be the tensor of features, shape (S, A, D).
        # We reshape F to (S*A, D) and new_logits to (S*A,) and solve
        # a least squares problem: minimize || F*w - new_logits ||^2.
        F_tensor = self.F(M)  # shape (S, A, D)
        S, A_num, D = F_tensor.shape
        X = F_tensor.reshape((S * A_num, D))
        y = new_logits.reshape((S * A_num,))
        
        # Solve for the new weights.
        new_weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.weights = new_weights
        
        # Return the updated performance.
        J = M.J(self.to_table(M))
        return J

    def n_gd_update(self, pi, M, learning_rate):
        """
        N Gradient Descent (N-GD) update.
        
        Compute the N gradient:
        For each state s:
            g_N(s) = sum_a [ pi(s,a)*A(s,a)*(e_a - pi(s)) ]
        where advantage is computed as:
            A = M.Q(pi) - M.V(pi)
        
        Then update the logits:
            q_new = ln(pi) + learning_rate * g_N,
        and recover the new policy via softmax.
        """
        S, A = pi.shape
        # d: discounted state distribution (vector of length M.S)
        d = M.d(pi)
        Q_vals = M.Q(pi)    # shape (S, A)
        V_vals = M.V(pi)    # shape (S,)
        A_mat = Q_vals - V_vals[:, None]  # Advantage matrix, shape (S, A)
        
        pi_new = np.zeros_like(pi)
        for s in range(S):
            # Current natural parameter (logits) computed from current policy
            q = np.log(pi[s, :] + 1e-8)
            g_N = np.zeros(A)
            for a in range(A):
                # One-hot vector for action a
                e_a = np.zeros(A)
                e_a[a] = 1.0
                g_N += d[s] * pi[s, a] * A_mat[s, a] * (e_a - pi[s, :])
            q_new = q + learning_rate * g_N
            pi_new[s, :] = softmax(q_new)
        return pi_new

    def m_gd_update(self, pi, M, learning_rate):
        """
        M Gradient Descent (M-GD) update.
        
        In the mean (probability) space, we use the M gradient:
            g_M(s,a) = d(s) * A(s,a),
        where advantage is A = M.Q(pi) - M.V(pi).
        
        Then we update the natural parameters by:
            q_new = ln(pi) + learning_rate * g_M,
        and recover the new policy via softmax.
        """
        S, A = pi.shape
        # d: discounted state distribution (vector of length M.S)
        d = M.d(pi)
        Q_vals = M.Q(pi)
        V_vals = M.V(pi)
        A_mat = Q_vals - V_vals[:, None]
        
        pi_new = np.zeros_like(pi)
        for s in range(S):
            q_nat = np.log(pi[s, :] + 1e-8)
            g_M = d[s] * A_mat[s, :]
            # g_M = A_mat[s, :]
            q_new = q_nat + learning_rate * g_M
            pi_new[s, :] = softmax(q_new)
        return pi_new

    def n_md_update(self, pi, M, learning_rate):
        """
        N Mirror Descent (N-MD) update.
        
        In the N space, using the REINFORCE gradient we have:
            g_N(s) = sum_a [ pi(s,a)*A(s,a)*(e_a - pi(s)) ].
        Then we update the natural parameter as:
            q_new = ln(pi) + learning_rate * F^{-1} * g_N,
        where for each state s,
            F = diag(pi(s)) - pi(s)pi(s)^T.
        Finally, recover the new policy via softmax.
        """
        S, A = pi.shape
        d = M.d(pi)
        Q_vals = M.Q(pi)
        V_vals = M.V(pi)
        A_mat = Q_vals - V_vals[:, None]
        
        pi_new = np.zeros_like(pi)
        for s in range(S):
            q = np.log(pi[s, :] + 1e-8)
            g_N = np.zeros(A)
            for a in range(A):
                e_a = np.zeros(A)
                e_a[a] = 1.0
                g_N += d[s] * pi[s, a] * A_mat[s, a] * (e_a - pi[s, :])
            # Compute per-state Fisher matrix: F = diag(pi) - pi*pi^T.
            ps = pi[s, :]
            F = np.diag(ps) - np.outer(ps, ps)
            F_inv = np.linalg.pinv(F)
            q_new = q + learning_rate * (F_inv @ g_N)
            pi_new[s, :] = softmax(q_new)
        return pi_new

    def m_md_update(self, pi, M, learning_rate):
        """
        M Mirror Descent (M-MD) update.
        
        In the M space, map probabilities to natural parameters:
            q_nat = ln(pi),
        compute the M gradient:
            g_M(s,a) = d(s) * A(s,a),
        then update:
            q_new = q_nat + learning_rate * F * g_M,
        where F = diag(pi) - pi*pi^T. Finally, recover the new policy via softmax.
        """
        S, A = pi.shape
        # d: discounted state distribution (vector of length M.S)
        d = M.d(pi)
        Q_vals = M.Q(pi)
        V_vals = M.V(pi)
        A_mat = Q_vals - V_vals[:, None]
        
        pi_new = np.zeros_like(pi)
        for s in range(S):
            q_nat = np.log(pi[s, :] + 1e-8)
            g_M = d[s] * A_mat[s, :]
            # g_M = A_mat[s, :]
            ps = pi[s, :]
            F = np.diag(ps) - np.outer(ps, ps)
            q_new = q_nat + learning_rate * (F @ g_M)
            pi_new[s, :] = softmax(q_new)
        return pi_new

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

    def compute_returns(self, rewards, gamma):
        """
        Compute discounted returns for an episode.
        rewards: list of rewards for the episode.
        Returns: list of discounted returns, one per time step.
        """
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        # Compute returns in reverse order:
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
        return returns

    def reinforce(self, pi, M, learning_rate, gamma, Q_star, V_star):
        """
        Implements the REINFORCE algorithm.
        
        Parameters:
        M            : The MDP environment with attributes:
                        - M.s0: initial state distribution.
                        - M.P: transition probabilities (S x A x S array).
                        - M.R: reward function (S x A x S array).
        pi           : Initial policy table (NumPy array of shape (S, A)).
        learning_rate: The step size (denoted by $\alpha$).
        gamma        : Discount factor.        
        Returns:
        Updated policy table (NumPy array of shape (S, A)).
        """
        S, A = pi.shape
        
        # Lists to store the trajectory.
        states = []
        actions = []
        rewards = []
        
        # Sample an episode.
        s = sample(M.s0)
        max_steps = 20
        
        for t in range(max_steps):
            states.append(s)
            # Sample an action from the current policy at state s.
            a = sample(pi[s])
            actions.append(a)
            # Sample next state and obtain reward.
            sp = sample(M.P[s, a, :])
            r = M.R[s, a, sp]
            rewards.append(r)
            s = sp
        
        # Compute the discounted returns for this episode.
        returns = self.compute_returns(rewards, gamma)

        v_star_values = []
        
        # For each time step in the episode, update the policy at state s.
        for t, s in enumerate(states):
            a = actions[t]
            G_t = returns[t]
            # v_star_values.append(V_star[s, a])


            # Current policy at state s: pi(s) (a vector of length A)
            # Compute the current natural parameters (logits)
            q = np.log(pi[s] + 1e-8)
            # Construct one-hot vector for action a.
            e = np.zeros(A)
            e[a] = 1.0
            # REINFORCE update in the logit space:
            # q_new = ln(pi(s)) + learning_rate * G_t * (e - pi(s))
            q_new = q + learning_rate * (G_t - M.V(pi)[s]) * (e - pi[s])
            # q_new = q + learning_rate * G_t * (e - pi[s])
            # Update the policy at state s.
            pi[s] = softmax(q_new)
                
        
        print(np.isclose(M.V(pi), V_star, atol = 0.1).all())
        # output =  f"v_star_values: {v_star_values}\n"
        # output += f"returns: {returns}\n"
        # output += f"---------------------\n\n"
        # print(output)
        return pi

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
