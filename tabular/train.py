import argparse
import numpy as np
import pylab as pl
import pandas as pd
from datetime import datetime  # Import datetime to create timestamps

from arsenal.viz import axman
from arsenal.maths import sample
from arsenal import iterview, Alphabet

from leach.tabular.src.policy import SoftmaxPolicy, NoRegretImitationLearning
from leach.tabular.src.random_mdp import GarnetMDP, FixToyMDP # , MazeMDP
from leach.tabular.src.qlearn import QLearner

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ToyMDP','GarnetMDP'], default='ToyMDP')
parser.add_argument('--episodes', type=int, default=50000) # 300000
parser.add_argument("--solvers",
                    nargs='+',
                    choices=['policy_gradient', 'natural_policy_gradient', 'policy_iteration', 'reinforce', 'qlearning'],
                    default=['natural_policy_gradient'])
parser.add_argument("--learning_rate", type=float, default=1e-4) # 1e-5
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
    # opt_Q = M.Q(opt)
    # Jopt = M.J(opt)

    data = []

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
            continue
        elif solver == 'policy_gradient':
            for t in iterview(range(args.episodes)):
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
        elif solver == 'natural_policy_gradient':
            for t in iterview(range(args.episodes)):
                # Compute the natural gradient update.
                J, nat_grad = π.natural_policy_gradient(M)
                # Update parameters using the natural gradient.
                π.update(-nat_grad, learning_rate=args.learning_rate)
                data.append(dict(
                    t = t,
                    solver = solver,
                    gamma = args.gamma,
                    J = M.J(π.to_table(M,F)),
                    Jopt = M.J(opt),
                ))
            π = π.to_table(M,F)
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
    with axman('RL Alg Results:'):
        # Group the data by solver, averaging every 100 data points for smoothing.
        for name, dd in df.groupby('solver'):
            # Drop the solver column and group by batches of 100 iterations
            dd = dd.drop(['solver'], axis=1).groupby(dd.index // 100).mean()
            xs = dd.t      # Episode number (or iteration)
            ys = dd.J      # Current policy's performance (expected return J)
            pl.plot(xs, ys, label=f'{name}')

        # Plot the optimal performance Jopt as a horizontal dashed line.
        # Since Jopt is constant across iterations, we extract the first value.
        jopt = df['Jopt'].iloc[0]
        pl.axhline(jopt, color='k', linestyle='--', label='Optimal Performance')

        # Label the axes.
        pl.xlabel("Episode Number")
        pl.ylabel("Expected Return (J)")
        pl.legend(loc='best')

    pl.ioff()
    pl.savefig(f'slearn_{timestamp}.png')

    # Save Grid with a timestamp added to the filename (only for ToyMDP)
    if args.env == 'ToyMDP':
        pi = π
        # Use the timestamp in both the title and the filename
        G.draw(None,
               V = convert(M.V(pi)),
               policy = convert(pi),
               relevance = convert(M.d(pi)),
               title = f'pg',
               filename= f'pg_{timestamp}.png')

if __name__ == '__main__':
    main()
