import numpy as np
import pylab as pl
from arsenal.maths import compare, onehot
from arsenal import colors, iterview
ok = colors.green % 'ok'

from leach.tabular.src.mdp import DiscountedMDP, FiniteHorizonMDP, MRP, dirichlet, random_MDP


def test_lp_solver(M):
    # primary testing strategy is to compare the linear programming solver (LP)
    # to the value iteration solver (VI). In addition to checking equivalence of
    # the policies found by each, we also compare equivalence of other
    # quantities found by the LP. The dual variables should be value functions
    # (equal to VI's). The primal variables should be the joint state-action
    # distribution of the policy.
    lp = M.solve_by_lp()
    vi = M.solve_by_value_iteration()
    pi = lp['policy']
    assert np.allclose(lp['policy'], vi['policy'])
    print('[lp-solver] policy', ok)

    # Objective value matches the solution found by VI.
    assert abs(lp['obj'] - vi['obj']) / abs(vi['obj']) < 0.01
    print('[lp-solver] objective value', ok)

    d = M.d(lp['policy'])
    assert (d >= 0).all() and abs(d.sum() - 1) < 1e-10, 'stationary distribution did not a valid distribution.'
    assert compare(lp['mu'], M.d(pi), verbose=False).max_err < 1e-5
    print('[lp-solver] stationary distribution', ok)

    #compare(vi['V'], lp['V']) #.show()
    assert np.allclose(vi['V'], lp['V'])
    print('[lp-solver] value function', ok)

    # Test the relationships between primal and dual LPs
    dlp = M.solve_by_dual_lp()
    assert np.allclose(dlp['policy'], lp['policy'])
    assert np.allclose(dlp['obj'], lp['obj'])
    assert np.allclose(dlp['V'], lp['V'])
    assert np.allclose(dlp['mu'], lp['mm'].flatten())
    print('[dual-lp-solver]', ok)


def test_potential_based_shaping(M0):
    S = M0.S; A = M0.A; s0 = M0.s0

    opt_pi = M0.solve_by_value_iteration()['policy']

    # generate a random potential function
    phi = np.random.uniform(-1, 1, size=S)

    M1 = M0.copy()    # use a copy!
    M1.apply_potential_based_shaping(phi)

    # Check that both methods found the same policy
    original = M0.solve_by_value_iteration()
    shaped = M1.solve_by_value_iteration()
    assert np.allclose(shaped['policy'], original['policy'])

    opt_pi = M0.solve_by_value_iteration()['policy']

    pi = dirichlet(S, A)

    v0 = M0.V(pi)
    v1 = M1.V(pi)

    # Corollary 2 of Ng et al. (1999).
    assert np.allclose(v0, v1 + phi)

    # Advantage is invariant to shaping.
    assert np.allclose(M0.Advantage(pi), M1.Advantage(pi))

    # The difference in J only depends on the initial state
    assert np.allclose(M0.J(pi), M1.J(pi) + s0.dot(phi))
    print('[potential-based shaping] relationship between expected values and value functions', ok)

    # shaping with the optimal value function
    # TODO: are there other interesting things to say about this setting?
    vstar = original['V']
    M2 = M0.copy()  # use a copy of R!
    M2.apply_potential_based_shaping(vstar)

    assert np.allclose(0, M2.V(opt_pi))  # optimal policy as V=0 everywhere and everything else in negative
    assert (M2.V(pi) <= 0).all()         # suboptimal policies have negative value everywhere

    # optimal policy in the "optimally shapped MDP" can be found with gamma=0!
    M2.gamma *= 0
    assert (M2.solve_by_value_iteration()['policy'] == opt_pi).all()

    # The optimal policy in M2 requires *zero* steps of lookahead (i.e., just
    # optimize immediate reward). The proof is pretty trivial.
    #
    # Given the V*-shaped reward R':
    #    R'[s,a,s'] def= R[s,a,s'] + γ V*[s'] - V*[s]
    #
    # R'[s,a] = sum_{s'} p(s' | s, a) * R'[s,a,s']
    #         = sum_{s'} p(s' | s, a) * (R[s,a,s'] + γ V*[s'] - V*[s])
    #         = A*(s,a)
    #
    # Acting greedily according to A*(s,a) is clearly optimal. Nonetheless, we
    # have an explict test below.
    assert np.allclose(M2.r_sa(), M0.Advantage(opt_pi))
    M2_r_sa = M2.r_sa()
    myopic_pi = (M2_r_sa == M2_r_sa.max(axis=1)[:,None]) * 1.0
    assert np.allclose(myopic_pi, opt_pi)

    print('[potential-based shaping] "optimal shaping"', ok)


def test_performance_difference_lemma_discounted(M):
    """
    Evaluate performance difference of `p` over `q` based on roll-outs from on
    `q` and roll-ins from `p`.
    """
    # Connection to performance-difference lemma.
    #
    # If we take ϕ(s) = Vq(s), the value function an arbitrary policy q,
    #
    #   R'(s,a,s') = R(s,a,s') + γ Vq(s') - Vq(s)
    #
    # And then take the expectation over s',
    #
    #   E_{s'}[ R'(s,a,s') ]
    #     = E_{s'}[ R(s,a,s') + γ Vq(s') - Vq(s) ]
    #     = E_{s'}[ R(s,a,s') + γ Vq(s')  ]  - Vq(s)
    #     = Qq(s,a) - Vq(s).
    #
    # We see that the effective reward function is the advantage.
    #
    # TODO: Now, the question is what does the action-value function look like after
    # shaping?
    #
    # TODO: There is some discussion in the Ng and Russel papers about the idealized
    # case of value function (p's value funciton not q's).
    #
    #   - I think it's quite simple. When p=q, the advantage is always
    #     zero. Therefore, variance is zero.
    #
    #   - V* is also an interesting case, which is closer to the SEARN case.

    p = dirichlet(M.S, M.A)
    q = dirichlet(M.S, M.A)

    dp = M.d(p)           # Roll-in with p
    Aq = M.Advantage(q)   # Roll-out with q
    # Accumulate advantages of p over q.
    z = 1/(1-M.gamma) * sum(dp[s] * p[s,:].dot(Aq[s,:]) for s in range(M.S))

    assert np.allclose(M.J(p) - M.J(q), z)
    print('[pd-lemma]', ok)


def test_finite_horizon():
    print()
    print('Finite-horizon tests:', ok)

    S = 10
    A = 3
    M = FiniteHorizonMDP(
        s0 = dirichlet(S),
        R = np.random.uniform(0,1,size=(S,A,S)),
        P = dirichlet(S,A,S),
        T = 20,
    )

    p = dirichlet(M.S, M.A)
    assert abs(M.d(p).sum() - M.T)/M.T < 1e-5

    test_pd_lemma_finite_horizon(M)


def test_pd_lemma_finite_horizon(M):
    """
    Evaluate performance difference of `p` over `q` based on roll-outs from on
    `q` and roll-ins from `p`.
    """
    p = dirichlet(M.S, M.A)
    q = dirichlet(M.S, M.A)

    Jq,Vq,Qq = M.value(q)   # Roll-out with q
    dp = M.d(p)             # Roll-in with p. Note that dp sums to T, not 1.
    #assert dp.sum() == M.T

    Jp,_,_ = M.value(p)     # Value p.
    # Accumulate advantages of p over q.
    z = 0.0
    for t in range(M.T):
        for s in range(M.S):
            A = p[s,:].dot(Qq[t,s,:]) - Vq[t,s]
            z += dp[t,s] * A
    assert np.allclose(Jp - Jq, z)
    print('[pd-lemma]', ok)


def test_simulator(M):
    from arsenal.maths import sample
    print('[test simulator]')
    assert np.allclose(M.P_with_reset().sum(axis=2), 1)
    pi = np.ones((M.S, M.A)) / M.A
    test_mrp_simulator(M.mrp(pi))


def test_mrp_simulator(M):
    assert np.allclose(M.P_with_reset().sum(axis=1), 1)

    class LearnModel:
        def __init__(self):
            self.A = np.zeros((M.S, M.S))

            self.M = np.zeros((M.S, M.S))
            self.N = np.zeros(M.S)
            self.R = np.zeros(M.S)
            self.b = np.zeros(M.S)
            self.t = 0

        def update(self, s, r, sp):
            self.R[s] += r
            self.N[s] += 1
            self.M[s,sp] += 1


            fs = onehot(s, M.S)
            fp = onehot(sp, M.S)
            self.A += np.outer(fs, fs - M.gamma*fp)   # xref:discounted features
#            self.A += np.outer(fs, fs - fp)          # xref:undiscounted features
            self.b += r * fs

            self.t += 1

        def estimates(self):
            P = self.M / self.N[:,None]
            R = self.R / self.N[:]
            assert np.allclose(P.sum(axis=1), 1)
            return P, R

    agent = LearnModel()

    from arsenal.viz import lc
    from scipy import linalg

    lc['mrp-sim'].yscale = 'log'
    lc['mrp-sim'].xscale = 'log'


    # LSTD matrix converges to the following.
    I = np.eye(M.S)
    d = M.d()
    A = np.diag(d) @ (I - M.P_with_reset())
#    A = np.diag(d) @ (I - M.gamma*M.P)

    b = M.R * d   # because we normalize by t not by N[s]

    V = M.V()

    for t, (s, r, sp) in enumerate(M.run(), start=1):
        r += np.random.normal(0, 1)
        agent.update(s, r, sp)
        if t % 1000 == 0:
            p, r = agent.estimates()

            # TODO: I've reached a stale mate between getting A to converge and
            # having an ill-condiitoned A v = b problem.  The issue is that `A`
            # our simulator already handles discounting to some extent. See
            # xref:discounted features and xref:undiscounted features.  The
            # ill-conditioning isn't for sampling, its from the definition
            # itself
#            lc['mrp-sim'].update(t, A = linalg.norm(A - agent.A/agent.t))
            lc['mrp-sim'].update(t, b = linalg.norm(b - agent.b/agent.t))

#            VV = linalg.solve(agent.A/agent.t, agent.b/agent.t)
#            lc['mrp-sim'].update(t, v = linalg.norm(V - VV))

            # stationary distribution seems to work.
#            lc['mrp-sim'].update(t, d = linalg.norm(d - agent.N/agent.t))

            # Transition and reward functions seem to work as well.
#            lc['mrp-sim'].update(t, p = linalg.norm(p - M.P_with_reset()))
#            lc['mrp-sim'].update(t, r = linalg.norm(r - M.R))



# TODO: resurrect tests
#def test_mdp_simulator():
#    class LearnModel:
#        def __init__(self):
#            self.N = np.zeros((M.S, M.A, M.S))
#            self.M = np.zeros((M.S, M.A))
#            self.R = np.zeros((M.S, M.A))
#            self.t = 0
#
#        def __call__(self, s):
#            return sample(pi[s])
#
#        def update(self, s, a, r, sp):
#            self.R[s,a] += r
#            self.N[s,a,sp] += 1
#            self.M[s,a] += 1
#            self.t += 1
#            return self.t <= 100000
#
#    agent = LearnModel()
#    M.run(agent)
#
#    P = agent.N
#    R = agent.R
#    for s in range(M.S):
#        for a in range(M.A):
#            P[s,a,:] /= agent.M[s,a]
#            R[s,a] /= agent.M[s,a]
#
#    #compare(M.P, P).show(title='transition function')
#    #compare(M.P_with_reset(), P).show(title='transition function')
#    compare(np.einsum('sap,sap->sa', M.P_with_reset(), M.R), R).show(title='reward function')
#
#    A = M.solve_by_value_iteration()['policy']
#
#    est = DiscountedMDP(M.s0, P, R, M.gamma)
#    B = est.solve_by_value_iteration()['policy']
#
#    compare(A, B).show('optimal policies')


def main():
    M = random_MDP(20, 5, 0.7)

    vi = M.solve_by_value_iteration()
    PI = M.solve_by_policy_iteration()
    assert np.allclose(vi['V'], PI['V'])
    assert np.allclose(vi['policy'], PI['policy'])
    assert np.allclose(vi['obj'], PI['obj'])
    print('policy iteration == value iteration', ok)

    π = vi['policy']
    d = M.d(π)
    assert (d >= 0).all() and abs(d.sum() - 1) < 1e-10, 'stationary distribution did not a valid distribution.'
    print('stationary distribution is valid', ok)

    J = M.J(π)
    assert abs(J - vi['obj']) / abs(J) < 0.01
    print('value of policy matches VI', ok)

    mrp = M.mrp(π)
    sv = mrp.successor_representation().dot(mrp.R)
    assert np.allclose(sv, vi['V'])
    print('successor representation', ok)

    v = M.V(π)
    assert np.allclose(v, vi['V'])
    print('value function', ok)

    Q = M.Q(π)
    assert np.allclose(v, np.einsum('sa,sa->s', Q, π))
    print('check V is average Q', ok)

    assert np.allclose(Q, M.Q_by_linalg(π))
    print('check Q by linalg', ok)

    test_stationary(mrp)
    test_lp_solver(M)
    test_performance_difference_lemma_discounted(M)
    test_potential_based_shaping(M)
    #test_simulator(M)   # TODO: create an automated test.


def random_mrp(S, gamma=0.3):
    # Randomly generate and MDP.
    return MRP(
        s0 = dirichlet(S),
        R = np.random.uniform(0,1,size=S),
        P = dirichlet(S, S),
        gamma = gamma,
    )


def test_J(M):
    J0 = M.J()

    j1 = M.V().dot(M.s0)
    j3 = M.R.dot(M.d()) / (1-M.gamma)

    # [2018-09-26 Wed] The following idea was tempting, but wrong! Here is where
    # my logic broke down: In the case of MDPs, we can use the performance
    # difference lemma (PD) to create a similar equation.  However, PD relates
    # the expected advantage function under a stationary distribution to the
    # difference of J's.  In the special case of a single PD of a policy verus
    # itself, we have that J'-J should be zero.  Note that the advantage of a
    # policy against itself is just the reward function.

    #j2 = M.V().dot(M.d()) / (1-M.gamma)  # <=== INCORRECT!


    assert np.allclose(J0, j1)
    #assert np.allclose(J0, j2), [J0, j2]
    assert np.allclose(J0, j3)


    # Test a single-state MRP
    # Sanity check: Why is there a 1/(1-γ) here?
    # if there is 1 state {
    #   rewards    = [r]
    #   stationary = [1]
    #   value      = r + γ value
    #              = r / (1-γ)
    #   J          = r / (1-γ)
    # }
    m1 = random_mrp(1)
    assert np.allclose(m1.J(), m1.R / (1-m1.gamma))

    print('[test J]', ok)


# Again I was trying to understand the error in "J = d * V"
#    S = 20
#    A = 2
#
#    # Randomly generate and MDP.
#    M = DiscountedMDP(
#        s0 = dirichlet(S),
#        R = np.random.uniform(0,1,size=(S,A,S)),
#        P = dirichlet(S,A,S),
#        gamma = 0.7,
#    )
#
#    data = []
#    for _ in range(100):
#
#        pi = dirichlet(S,A)
#
#        j1 = M.J(pi)
#        j2 = M.V(pi).dot(M.s0)
#
#        #j3 = M.V(pi).dot(M.d(pi))
#
#        #Adv = M.Advantage(1-pi)
#        Q = M.Q(1-pi)
#        V = M.V(1-pi)
#
#        d = M.d(pi)
#        #j3 = sum(Adv[s,a] * pi[s,a] * d[s] for s in range(S) for a in range(A))
#        #j3 = sum((Q[s,a] - V[s]) * pi[s,a] * d[s] for s in range(S) for a in range(A))
#        #j3 = (sum(Q[s,a] * pi[s,a] * d[s] for s in range(S) for a in range(A))
#        #      - sum(V[s] * pi[s,a] * d[s] for s in range(S) for a in range(A)))
#
#        j3 = (sum(Q[s,a] * pi[s,a] * d[s] for s in range(S) for a in range(A))
#              - sum(V[s] * d[s] for s in range(S)))
#
#        print()
#        print(j3 / (1-M.gamma))
#        print(M.J(pi) - M.J(1-pi))
#
#        #j4 = M.R.dot(M.d(pi)) / (1-M.gamma)
#
#        data.append({'j1': j1, 'j2': j2, 'j3': j3})
#
#    import pandas as pd
#    df = pd.DataFrame(data)
#    compare(df.j1, df.j3).show()


def test_stationary(M):
    print('[test stationary]')
    d1 = M.d()
    d2 = M.d_via_eigen()
    assert compare(d1, d2).max_relative_error < 1e-5

    J0 = M.J()

    if 1:
        p = np.zeros(M.S)
        J = 0.0
        N = 10_000
        err = []
        for t, [s, r, _] in zip(iterview(range(1,1+N)), M.run()):
            p[s] += 1
            J += r / (1-M.gamma)   # TODO: is this like an importance sampling correction for early stopping?

            if t % 5000 == 0:
                err.append([t, abs(J/t - J0)])

        p /= N
        J /= N
        compare(d1, p) #.show()
        Jmax = M.R.max() / (1-M.gamma)
        assert abs(J - M.J()) < Jmax/np.sqrt(N)

        if 0:
            # Error decays at a rate of 1/sqrt(N)
            ns, err = np.array(err).T
            bs = Jmax/np.sqrt(ns)
            pl.loglog(ns, bs)
            pl.loglog(ns, err)
            pl.show()


# TODO: unfinished
# Singh & Yee, 1993 "An Upper Bound on the Loss from Approximate Optimal-Value Functions"
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.8149&rep=rep1&type=pdf
def test_SinghYee_VFA_bound(M):

    opt = M.solve_by_value_iteration()['policy']
    Vopt = M.V(opt)
    Jopt = M.J(opt)

    data = []
    for _ in iterview(range(10000)):

        vfa = np.random.uniform(-10,10,size=M.S)

        # let epsilon be the error in a value function approximator
        epsilon = np.abs(vfa - Vopt).max()

        # Note: this requires one-step lookahead knowledge
        [_, greedy] = M.B(vfa)
        actual = Jopt - M.J(greedy)

        assert actual >= 0

        # it follows that the greedy policy wrt `vfa` has bounded error
        bound = 2 * M.gamma * epsilon / (1 - M.gamma)

        # There is a simple extension for error in estimate of R

        # TODO: there is a related bound for estimation error in Q. See corollary 2.
        # TODO: there is also an extension in terms of one-step error

        #print(epsilon, actual, bound)

        data.append({'epsilon': epsilon,
                     'actual': actual,
                     'bound': bound})

    import pandas as pd
    df = pd.DataFrame(data).sort_values('epsilon')

    pl.yscale('log'); pl.xscale('log')
    pl.scatter(df.epsilon, df.actual, c='b')
    pl.plot(df.epsilon, df.bound, c='r')
    pl.xlabel('epsilon')
    pl.ylabel('J(opt) - J(greedy wrt est)')
    pl.show()


if __name__ == '__main__':
    main()
    test_finite_horizon()
    #test_SinghYee_VFA_bound(random_MDP(20, 5, 0.7))
