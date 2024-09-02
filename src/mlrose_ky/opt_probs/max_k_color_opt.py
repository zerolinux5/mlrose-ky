"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import networkx as nx
import numpy as np

from mlrose_ky.algorithms.crossovers import UniformCrossover
from mlrose_ky.algorithms.mutators import ChangeOneMutator
from mlrose_ky.fitness import MaxKColor
from mlrose_ky.opt_probs.discrete_opt import DiscreteOpt


class MaxKColorOpt(DiscreteOpt):
    def __init__(self, edges=None, length=None, fitness_fn=None, maximize=False, max_colors=None, crossover=None, mutator=None, source_graph=None):
        if (fitness_fn is None) and (edges is None):
            raise Exception("fitness_fn or edges must be specified.")

        if length is None:
            if fitness_fn is None:
                length = len(edges)
            else:
                length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = MaxKColor(edges, maximize)

        if length == 1:
            self.max_val = 1
            self.source_graph = nx.Graph()
            fitness_fn.set_graph(self.source_graph)
            self.stop_fitness = 0 if maximize else 0
            super().__init__(length, fitness_fn, maximize, 1, crossover, mutator)
            self.set_state(np.array([0]))
            return

        # Create or update the graph to include all nodes up to `length`
        if source_graph is None:
            g = nx.Graph()
            g.add_edges_from(edges)
            # Ensure all nodes up to `length` are added
            for i in range(length):
                if i not in g:
                    g.add_node(i)
            self.source_graph = g
        else:
            self.source_graph = source_graph

        self.stop_fitness = self.source_graph.number_of_edges() if maximize else 0

        fitness_fn.set_graph(self.source_graph)

        if max_colors is None:
            total_neighbor_count = [len([*self.source_graph.neighbors(n)]) for n in range(length)]
            if total_neighbor_count:
                max_colors = 1 + max(total_neighbor_count)
            else:
                max_colors = 1  # Default to 1 when there are no edges
        self.max_val = max_colors

        crossover = UniformCrossover(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_colors, crossover, mutator)

        state = np.random.randint(max_colors, size=self.length)
        np.random.shuffle(state)
        self.set_state(state)

    def can_stop(self):
        return int(self.get_fitness()) == self.stop_fitness
