# -*- coding: utf-8 -*-
"""
Grid world environment
"""

import pylab as pl
import numpy as np
from matplotlib.table import Table
import os

from leach.tabular.src.mdp import DiscountedMDP
from arsenal import Alphabet
from arsenal.viz import update_ax

class Action(object):
    def __init__(self, name, dx, dy):
        self.name = name
        self.dx = dx
        self.dy = dy
    def __iter__(self):
        return iter((self.dx, self.dy))
    def __repr__(self):
        return self.name


class GridWorld(object):
    """A two-dimensional grid MDP.  All you have to do is specify the grid as a list
    of lists of rewards; use None for an obstacle. Also, you should specify the
    terminal states.  An action is an (x, y) unit vector; e.g. (1, 0) means move
    east.

    """
    EMPTY, WALL, START, GOAL, QM, PIT = np.array(['0','W','S','G', '?', 'P'], dtype='|S1')

    def __init__(self, mapname=None, grid=None):
        # Load Map
        __location__ = os.path.dirname((os.path.abspath(__file__)))
        default_map_dir = os.path.join(__location__, "maps")

        self.ax = None
        self.A = [
            Action('⬆',0,-1), # up     b/c 0,0 is upper left
            Action('⬇',0,1),  # down
            Action('⬅',-1,0), # left
            Action('➡',1,0),  # right
        ]

        self.states = set()
        self.reward = {}

        if grid is None:
            self.grid = grid = np.loadtxt(os.path.join(default_map_dir, mapname), dtype='|S1')
        else:
            self.grid = np.array(grid)
            # TODO: Fix this hack
            self.START = 'S'
            self.WALL = 'W'
            self.GOAL = 'G'
            self.PIT = 'P'
            self.EMPTY = '0'
            self.QM = '?'


        [self.rows, self.cols] = self.grid.shape

        # Parse Map
        self.initial_state = np.argwhere(self.grid == self.START)[0]
        self.initial_state = tuple(self.initial_state)

        self.terminals = np.argwhere(self.grid == self.GOAL)[0]
        self.terminals = [tuple(self.terminals)]

        # Convert Numpy bytes to int
        new_grid = np.zeros(self.grid.shape)

        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] != self.WALL:
                    s = (r, c)
                    self.states.add(s)

                    if grid[r,c] == self.PIT: value = -1.0
                    elif grid[r,c] == self.GOAL: value = 2.0
                    elif grid[r,c] == self.START: value = 0.0
                    elif grid[r,c] == self.EMPTY: value = 0.0
                    else: raise Exception("Unknown grid attribute: ", grid[r,c])

                    self.reward[s] = value
                    new_grid[r,c] = value
                else:
                    new_grid[r,c] = None

        # Hack to convert np.nan to None
        # (Line 87 results in np.nan instead of None)
        new_grid = np.where(np.isnan(new_grid), None, new_grid)
        self.grid = new_grid.copy()

    def encode(self):
        # Encode problem into an MDP so we can call its solver.
        Si = Alphabet()
        Ai = Alphabet()
        s0 = np.zeros(len(self.states))
        s0[Si[self.initial_state]] = 1
        P = np.zeros((len(self.states), len(self.A), len(self.states)))
        R = np.zeros((len(self.states), len(self.A), len(self.states)))

        for s in self.states:
            #row, col = s
            si = Si[s]
            for a in self.actions(s):
                ai = Ai[a]
                sp, r = self.simulate(s, a)
                spi = Si[sp]
                P[si,ai,spi] = 1    # deterministic environment
                R[si,ai,spi] = r

        #mdp = DiscountedMDP(s0, P, R, gamma=0.7)
        Si.freeze(); Ai.freeze()
        #return mdp, Si, Ai
        return (s0, P, R), Si, Ai

    def actions(self, _):
        return self.A

    def simulate(self, s, a):
        if s in self.terminals:
            return self.initial_state, 0.0
        dx, dy = a
        sp = (s[0] + dy, s[1] + dx)
        if sp in self.states:
            sp = sp
            r = self.reward[sp]
        else:
            sp = s
            r = 0
#            r = -0.05   # negative reward for crashing into a wall.
        return sp, r  # stay in same state if we hit a wall

    def draw(self, current_state, V, policy, c=None, relevance=None, title=None, ax=None, filename=None):
        "Render environment"

        if ax is None:
            self.ax = ax = self.ax or pl.figure(frameon=False).add_subplot(111, aspect='equal')

        grid = self.grid
        nrows, ncols = grid.shape

        with update_ax(ax):
            #ax.figure.canvas.toolbar.hide()
            ax.set_axis_off()

            scale = 1/max(nrows,ncols)
            #tb = Table(ax, loc=(0,0), bbox=[0,0,nrows*scale,ncols*scale])
            tb = Table(ax, loc=(0,0), bbox=[0,0,1,1])
            ax.add_table(tb)

            dots = []
            width, height = 1, 1
            for x in range(nrows):
                for y in range(ncols):
                    r = grid[x,y]
                    if r == None: color = 'black'
                    elif r == 0:  color = 'white'
                    elif r > 0:   color = 'green'
                    else:         color = 'red'
                    if r is not None and (x,y) not in self.terminals:
                        dots.append((x,y))
                    tb.add_cell(x, y, width, height, #text=s, loc='center',
                                facecolor = color)

            ax.figure.canvas.draw()   # need to run draw to define cell bboxes below.

            if title: ax.set_title(title)

            for s in dots:
                p = tb._cells[s].properties()['bbox']
                p = (p.p0 + p.p1)/2

                # make circle area proportional to on how often the agent visits
                # that state and color that state according to it's value
                # function.
                c = pl.cm.Blues(V[s])
                r = relevance[s]*len(self.states) if relevance is not None else 0.5#/len(self.states)

                circle = pl.Circle(p, fc=c, radius=0.2*scale*np.sqrt(r), linewidth=0)
                ax.add_patch(circle)

                if policy is not None:
                    A = tb._cells[s].properties()['bbox']
                    x,y = (A.p0 + A.p1)/2
                    for i, a in enumerate(self.A):
                        p = policy[s][i]
                        if p > 0:
                            (dx,dy) = a
                            ax.arrow(x, y, dx/40 * p, -dy/40 * p, color='k', width=.005 * p)

            if current_state is not None:
                a = tb._cells[current_state].properties()['bbox']
                a = (a.p0 + a.p1)/2
                circle = pl.Circle(a, radius=0.1*scale, fc='y', linewidth=0)
                ax.add_patch(circle)

            ax.figure.tight_layout()

        self.ax.figure.savefig(filename)
