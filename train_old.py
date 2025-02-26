import argparse
import numpy as np
import pylab as pl
import pandas as pd
import pdb
from datetime import datetime  # Import datetime to create timestamps

from arsenal.viz import axman
from arsenal.maths import sample
from arsenal import iterview, Alphabet

from leach.tabular.src.policy import SoftmaxPolicy, NoRegretImitationLearning
from leach.tabular.src.random_mdp import GarnetMDP, FixToyMDP # , MazeMDP
from leach.tabular.src.qlearn import QLearner

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ToyMDP','GarnetMDP'], default='ToyMDP')
parser.add_argument('--episodes', type=int, default=1000) # 300000
parser.add_argument("--solvers",
                    nargs='+',
                    choices=['policy_gradient', 'natural_policy_gradient', 'm_mirror_descent', 'n_mirror_descent', 'policy_iteration', 'reinforce', 'qlearning'],
                    default=['reinforce'])
parser.add_argument("--learning_rate", type=float, default=0.01) # 1e-5
parser.add_argument("--gamma", type=float, default=.99)
args = parser.parse_args()

def main():
    Si = None
    if args.env =='GarnetMDP':
        M, features, D = GarnetMDP()(S=50, A=15, b=5, gamma=args.gamma, H=20)
    elif args.env == 'ToyMDP':
        mdp = FixToyMDP()
        M, features, D = mdp(mapname="4x7.txt", gamma=args.gamma)
        Si, G, convert = mdp.Si, mdp.G, mdp.convert


    F = np.zeros((M.S, M.A, D))
    for s in range(M.S):
        for a in range(M.A):
            F[s,a,:] = features(s,a)

    # Create Optimal Policy for setting
    opt = M.solve_by_value_iteration()['policy']
    V_star = M.solve_by_value_iteration()['V']
    # opt_Q = M.Q(opt)
    # Jopt = M.J(opt)

    data = []
    final_policies = {}
    for solver in args.solvers:  # Remove iterview here for solver in iterview(args.solvers):
        π = SoftmaxPolicy(M.A, D, features)
        if solver == "policy_iteration":
            # Create Optimal Policy for setting
            π = M.solve_by_policy_iteration()['policy']
            data.append(dict(
                    t = args.episodes,
                    solver = solver,
                    gamma = args.gamma,
                    J = M.J(π),
                    Jopt = M.J(opt),
            ))
        elif solver == 'reinforce':
            pi = π.to_table(M,F)
            for t in range(args.episodes):
                buffer = []
                s = sample(M.s0)
                # pdb.set_trace()
                steps = 20

                # Sampling experiences
                for _ in range(steps):
                    a = sample(pi[s])
                    sp = sample(M.P[s,a,:])
                    #r = M.R[s,a,sp]
                    #q.update(s, a, r, sp)
                    s = sp
                    buffer.append((s,a))
                
                # Sampling done, doing update using the buffer
                pi = π.reinforce(pi, M, learning_rate=args.learning_rate, buffer=buffer)

                J = M.J(pi)
                data.append(dict(t=t, solver=solver, J=J, Jopt=M.J(opt)))
        elif solver == 'policy_gradient':
            for t in iterview(range(args.episodes)):
            # for t in range(args.episodes):
                [_, g] = π.policy_gradient(M)
                π.update(-g, learning_rate=args.learning_rate)

                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    J = M.J(π.to_table(M,F)),
                    Jopt = M.J(opt),
                ))

            π = π.to_table(M,F)
            final_policies[solver] = π
        elif solver == 'natural_policy_gradient':
            # for t in iterview(range(args.episodes)):
            for t in range(args.episodes):
                # Compute the natural gradient update.
                π_table = π.to_table(M,F)
                J, nat_grad = π.natural_policy_gradient(M, π_table)
                # Update parameters using the natural gradient.
                π_table += args.learning_rate * -nat_grad
                # π.update(-nat_grad, learning_rate=learning_rate)
                w_norm = np.linalg.norm(π.weights)
                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    # J = M.J(π.to_table(M,F)),
                    J = M.J(π_table),
                    Jopt = M.J(opt),
                    w_norm = w_norm,
                ))
            π = π.to_table(M,F)
            final_policies[solver] = π
        elif solver == 'm_mirror_descent':
            for t in iterview(range(args.episodes)):
                J = π.mirror_descent_update(M, learning_rate=args.learning_rate)
                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    J = J,
                    Jopt = M.J(opt),
                    w_norm = np.linalg.norm(π.weights),
                ))
            π = π.to_table(M, F)
        elif solver == 'n_mirror_descent':
            for t in iterview(range(args.episodes)):
                J = π.n_mirror_descent_update(M, learning_rate=args.learning_rate)
                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    J = J,
                    Jopt = M.J(opt),
                    w_norm = np.linalg.norm(π.weights),
                ))
            π = π.to_table(M, F)
        elif solver == 'dual_mirror_descent':
            for t in iterview(range(args.episodes)):
                J = π.dual_mirror_descent_update(M, learning_rate=args.learning_rate)
                # Log the ℓ2-norm of the current weights
                w_norm = np.linalg.norm(π.weights)
                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    J = J,
                    Jopt = M.J(opt),
                    w_norm = w_norm,  # log weight norm
                ))
            π = π.to_table(M, F)
            final_policies[solver] = π
        elif solver == 'qlearning':
            q = QLearner(M, Si=Si)
            q.epsilon = 0.1
            q.alpha = 0.1
            steps = 100 # TODO: Do not hard-code Q-learning

            for t in iterview(range(args.episodes)):
                s = sample(M.s0)
                for _ in range(steps):
                    a = q(s)
                    sp = sample(M.P[s,a,:])
                    r = M.R[s,a,sp]
                    q.update(s, a, r, sp)
                    s = sp

                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    J = M.J(q.to_table()),
                    Jopt = M.J(opt),
                ))

            π = q.to_table()

    # Create a timestamp string, e.g., "20250210_153045"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = pd.DataFrame(data)
    # with axman('RL Alg Results:'):
    #     # Group the data by solver, averaging every 100 data points for smoothing.
    #     for name, dd in df.groupby('solver'):
    #         # Drop the solver column and group by batches of 100 iterations
    #         dd = dd.drop(['solver'], axis=1).groupby(dd.index // 100).mean()
    #         xs = dd.t      # Episode number (or iteration)
    #         ys = dd.J      # Current policy's performance (expected return J)
    #         pl.plot(xs, ys, label=f'{name}')

    #     # Plot the optimal performance Jopt as a horizontal dashed line.
    #     # Since Jopt is constant across iterations, we extract the first value.
    #     jopt = df['Jopt'].iloc[0]
    #     pl.axhline(jopt, color='k', linestyle='--', label='Optimal Performance')

    #     # Label the axes.
    #     pl.xlabel("Episode Number")
    #     pl.ylabel("Expected Return (J)")
    #     pl.legend(loc='best')

    # pl.ioff()

    # Create two subplots: one for performance, one for weight norm.
    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12, 5))

    # Group by solver for performance plot.
    for name, dd in df.groupby('solver'):
        # Smooth the data by grouping every 100 iterations (if desired)
        dd_perf = dd.drop(['solver'], axis=1).groupby(dd.index // 100).mean()
        ax1.plot(dd_perf.t, dd_perf.J, label=name)
    ax1.axhline(df['Jopt'].iloc[0], color='k', linestyle='--', label='Optimal Performance')
    ax1.set_xlabel("Episode Number")
    ax1.set_ylabel("Expected Return (J)")
    ax1.legend(loc='best')

    # Group by solver for weight evolution (using weight norm).
    for name, dd in df.groupby('solver'):
        dd_weights = dd.drop(['solver'], axis=1).groupby(dd.index // 100).mean()
        ax2.plot(dd_weights.t, dd_weights.w_norm, label=name)
    ax2.set_xlabel("Episode Number")
    ax2.set_ylabel("Weight Norm")
    ax2.legend(loc='best')

    fig.suptitle("Learning Performance and Weight Evolution")
    pl.tight_layout()
    pl.savefig(f'./output/slearn_{timestamp}.png')

    # Save Grid for each solver if using ToyMDP.
    if args.env == 'ToyMDP':
        for solver, pi in final_policies.items():
            # Create a filename with the solver name (and possibly a timestamp)
            filename = f'./output/pg_{solver}_{timestamp}.png'
            title = f'Learned Policy by {solver}'
            # Draw the grid using the provided conversion functions:
            G.draw(None,
                V = convert(M.V(pi)),
                policy = convert(pi),
                relevance = convert(M.d(pi)),
                title = title,
                filename = filename)

if __name__ == '__main__':
    main()
