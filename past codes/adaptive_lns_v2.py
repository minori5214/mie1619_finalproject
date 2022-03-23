from ALNS.alns import ALNS, State
from ALNS.alns.criteria import HillClimbing, SimulatedAnnealing

import copy
import itertools

import numpy as np

import networkx as nx

import tsplib95
import tsplib95.distances as distances

import matplotlib.pyplot as plt

SEED = 9876

#data = tsplib95.load('./instances/xqf131.tsp')

# These we will use in our representation of a TSP problem: a list of
# (city, coord)-tuples.
#cities = [(city, tuple(coord)) for city, coord in data.node_coords.items()]

#solution = tsplib95.load('./instances/xqf131.opt.tour')
#optimal = data.trace_tours(solution.tours)[0]

#print('Total optimal tour length is {0}.'.format(optimal))

class Instance_ALNS():
    def __init__(self, path):
        self.data = tsplib95.load(path)
        self.N = self.data.dimension

        if self.data.edge_weight_type == 'EXPLICIT' and \
            self.data.edge_weight_format == 'LOWER_DIAG_ROW':
            raise NotImplementedError
        elif self.data.edge_weight_type == 'EUC_2D':
            self.cities = [(city, tuple(coord)) for city, coord in self.data.node_coords.items()]
    
    def optimal_tour(self, opt_path):
        self.opt_path = opt_path
        solution = tsplib95.load(self.opt_path)
        tour = [[x-1 for x in solution.tours[0]]]
        print("Optimal tour: ", tour[0])

        optimal = self.data.trace_tours(solution.tours)[0]
        print('Total optimal tour length is {0}.'.format(optimal))

    def draw_graph(self, graph, only_nodes=False):
        """
        Helper method for drawing TSP (tour) graphs.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if only_nodes:
            nx.draw_networkx_nodes(graph, self.data.node_coords, node_size=25, ax=ax)
        else:
            nx.draw_networkx(graph, self.data.node_coords, node_size=25, with_labels=False, ax=ax)
        
        plt.show()
        plt.close()

#draw_graph(data.get_graph(), True)

class TspState(State):
    """
    Solution class for the TSP problem. It has two data members, nodes, and edges.
    nodes is a list of node tuples: (id, coords). The edges data member, then, is
    a mapping from each node to their only outgoing node.
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths,
        using the rounded Euclidean norm.
        """
        return sum(distances.euclidean(node[1], self.edges[node][1])
                   for node in self.nodes)
    
    def to_graph(self):
        """
        NetworkX helper method.
        """
        graph = nx.Graph()

        for node, coord in self.nodes:
            graph.add_node(node, pos=coord)

        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from[0], node_to[0])

        return graph

class ALNS_Agent():
    def __init__(self, state, degree_of_destruction=0.25, criterion='HillClimbing'):
        self.state = state
        self.degree_of_destruction = degree_of_destruction

        if criterion=='HillClimbing':
            self.criterion = HillClimbing()
        elif criterion=='SimulatedAnnealing':
            self.criterion = SimulatedAnnealing()

        self.random_state = np.random.RandomState(SEED)
        self.initial_solution = self.greedy_repair(state, self.random_state)
        print("Initial solution objective is {0}.".format(self.initial_solution.objective()))

    def edges_to_remove(self, state):
        return int(len(state.edges) * self.degree_of_destruction)

    def worst_removal(self, current, random_state):
        """
        Worst removal iteratively removes the 'worst' edges, that is,
        those edges that have the largest distance.
        """
        destroyed = current.copy()

        worst_edges = sorted(destroyed.nodes,
                            key=lambda node: distances.euclidean(node[1],
                                                                destroyed.edges[node][1]))

        for idx in range(self.edges_to_remove(current)):
            del destroyed.edges[worst_edges[-idx -1]]

        return destroyed

    def path_removal(self, current, random_state):
        """
        Removes an entire consecutive subpath, that is, a series of
        contiguous edges.
        """
        destroyed = current.copy()
        
        node_idx = random_state.choice(len(destroyed.nodes))
        node = destroyed.nodes[node_idx]
        
        for _ in range(self.edges_to_remove(current)):
            node = destroyed.edges.pop(node)

        return destroyed

    def random_removal(self, current, random_state):
        """
        Random removal iteratively removes random edges.
        """
        destroyed = current.copy()
        
        for idx in random_state.choice(len(destroyed.nodes),
                                    self.edges_to_remove(current),
                                    replace=False):
            del destroyed.edges[destroyed.nodes[idx]]

        return destroyed

    def would_form_subcycle(self, from_node, to_node, state):
        """
        Ensures the proposed solution would not result in a cycle smaller
        than the entire set of nodes. Notice the offsets: we do not count
        the current node under consideration, as it cannot yet be part of
        a cycle.
        """
        for step in range(1, len(state.nodes)):
            if to_node not in state.edges:
                return False

            to_node = state.edges[to_node]
            
            if from_node == to_node and step != len(state.nodes) - 1:
                return True

        return False

    def greedy_repair(self, current, random_state):
        """
        Greedily repairs a tour, stitching up nodes that are not departed
        with those not visited.
        """
        visited = set(current.edges.values())
    
        # This kind of randomness ensures we do not cycle between the same
        # destroy and repair steps every time.
        shuffled_idcs = random_state.permutation(len(current.nodes))
        nodes = [current.nodes[idx] for idx in shuffled_idcs]

        while len(current.edges) != len(current.nodes):
            node = next(node for node in nodes 
                        if node not in current.edges)

            # Computes all nodes that have not currently been visited,
            # that is, those that this node might visit. This should
            # not result in a subcycle, as that would violate the TSP
            # constraints.
            unvisited = {other for other in current.nodes
                        if other != node
                        if other not in visited
                        if not self.would_form_subcycle(node, other, current)}

            # Closest visitable node.
            nearest = min(unvisited,
                        key=lambda other: distances.euclidean(node[1], other[1]))

            current.edges[node] = nearest
            visited.add(nearest)

        return current
    
    def build(self):
        self.alns = ALNS(self.random_state)

        self.alns.add_destroy_operator(self.random_removal)
        self.alns.add_destroy_operator(self.path_removal)
        self.alns.add_destroy_operator(self.worst_removal)

        self.alns.add_repair_operator(self.greedy_repair)
    
    def run(self, time_limit=10):
        result = self.alns.iterate(self.initial_solution, [3, 2, 1, 0.5], 0.8, self.criterion,
                      iterations=5000, collect_stats=True, time_limit=time_limit)
        return result

if __name__ == "__main__":

    instance = Instance_ALNS('./instances/xqf131.tsp')
    instance.optimal_tour('./instances/xqf131.opt.tour')
    state = TspState(instance.cities, {})

    alns_agent = ALNS_Agent(state)
    alns_agent.build()
    result = alns_agent.run()

    solution = result.best_state

    objective = solution.objective()

    print('Best heuristic objective is {0}.'.format(objective))
    #print('This is {0:.1f}% worse than the optimal solution, which is {1}.'
    #      .format(100 * (objective - optimal) / optimal, optimal))

    _, ax = plt.subplots(figsize=(12, 6))
    result.plot_objectives(ax=ax, lw=2)
    plt.show()
    plt.close()

    figure = plt.figure("operator_counts", figsize=(14, 6))
    figure.subplots_adjust(bottom=0.15, hspace=.5)
    result.plot_operator_counts(figure=figure, title="Operator diagnostics", legend=["Best", "Better", "Accepted"])

    plt.show()
    plt.close()

    instance.draw_graph(solution.to_graph())