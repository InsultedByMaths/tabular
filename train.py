import argparse
import numpy as np
import pylab as pl
import pandas as pd
import pdb
from datetime import datetime  # Import datetime to create timestamps
import os

from arsenal import iterview
from arsenal.maths import sample
from leach.tabular.src.policy import SoftmaxPolicy, NoRegretImitationLearning
from leach.tabular.src.random_mdp import FixToyMDP, GarnetMDP

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ToyMDP','GarnetMDP'], default='ToyMDP')
parser.add_argument('--episodes', type=int, default=5000)
parser.add_argument("--solvers", nargs='+',
                    choices=['n_gd','m_gd','n_md','m_md','reinforce'],
                    # default=['reinforce'])
                    default=['n_gd', 'm_gd', 'n_md', 'm_md'])
parser.add_argument("--learning_rate", type=float, default=0.3)
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()

def main():
    # Initialize the MDP.
    if args.env == 'ToyMDP':
        mdp = FixToyMDP()
        M, features, D = mdp(mapname="4x7.txt", gamma=args.gamma)
        Si, G, convert = mdp.Si, mdp.G, mdp.convert
    else:
        M, features, D = GarnetMDP()(S=50, A=15, b=5, gamma=args.gamma, H=20)

    F = np.zeros((M.S, M.A, D))
    for s in range(M.S):
        for a in range(M.A):
            F[s,a,:] = features(s,a)
    
    # Compute an optimal policy for reference.
    opt = M.solve_by_value_iteration()['policy']
    V_star = M.V(opt)
    Q_star = M.Q(opt)
    
    data = []
    final_policies = {}
    for solver in args.solvers:
        # Initialize a uniform policy table (shape (S, A)).
        π = SoftmaxPolicy(M.A, D, features)
        pi = π.to_table(M,F)

        for t in iterview(range(args.episodes)):
        # for t in range(args.episodes):        
            # Update the policy using the chosen method.
            if solver == 'n_gd':
                pi = π.n_gd_update(pi, M, learning_rate=args.learning_rate)
            elif solver == 'm_gd':
                pi = π.m_gd_update(pi, M, learning_rate=args.learning_rate)
            elif solver == 'n_md':
                pi = π.n_md_update(pi, M, learning_rate=args.learning_rate)
            elif solver == 'm_md':
                pi = π.m_md_update(pi, M, learning_rate=args.learning_rate)
            elif solver == 'reinforce':
                pi = π.reinforce(pi, M, learning_rate=args.learning_rate, gamma=args.gamma, Q_star=Q_star, V_star=V_star)
            
            J = M.J(pi)
            data.append(dict(t=t, solver=solver, J=J, Jopt=M.J(opt)))

        final_policies[solver] = pi
    # Create a timestamp string, e.g., "20250210_153045"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_dir = f"./output/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # Plot learning curves.
    df = pd.DataFrame(data)
    pl.figure()
    for name, dd in df.groupby('solver'):
        dd_grouped = dd.groupby(dd.index // 100).mean(numeric_only=True)
        if name in ['n_md', 'm_md']:
            ls = ':'       # dotted line for MD methods
            lw = 2.5       # thicker line
        else:
            ls = '-'       # solid line for others
            lw = 1.5       # default line width
        pl.plot(dd_grouped.t, dd_grouped.J, label=name, linestyle=ls, linewidth=lw)
    pl.axhline(df['Jopt'].iloc[0], color='k', linestyle='--', label='Optimal')
    pl.xlabel("Episode Number")
    pl.ylabel("Expected Return (J)")
    pl.legend(loc='best')
    pl.savefig(f"./output/{timestamp}/learning_curves.png")
    
    # If using ToyMDP, visualize the final policy.
    if args.env == 'ToyMDP':
        for solver, pi in final_policies.items():
            filename = f"./output/{timestamp}/policy_{solver}.png"
            title = f"Final Policy by {solver}"
            G.draw(None,
                    V=convert(M.V(pi)),
                    policy=convert(pi),
                    relevance=convert(M.d(pi)),
                    title=title,
                    filename=filename)

if __name__ == '__main__':
    main()