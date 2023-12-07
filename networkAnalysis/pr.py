import random
from typing import Set, List

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


class Node:
    def __init__(self, node_id, node_type):
        self.node_id: str = node_id
        self.node_type: str = node_type
        self.keys: Set[str] = set()
        self.neighbours: Set[Node] = set()
        self.compromised = False

    def add_neighbour(self, node: 'Node'):
        self.neighbours.add(node)

    def set_keys(self, keys: Set[str]):
        self.keys = keys

    def set_compromised(self, is_compromised):
        self.compromised = is_compromised

    def empty_keys(self):
        self.keys = set()

    def __repr__(self):
        return f"Node({self.node_id}, {self.node_type}, {self.keys}, is_compromised={self.compromised})"


class Graph:
    def __init__(self, s_count, i_count, d_count):
        self.nodes: Set[Node] = set()
        self.counters = {"source": 0, "intermediate": 0, "sink": 0}
        self.prefixes = {"source": "S", "intermediate": "I", "sink": "D"}

        for _ in range(s_count):
            self.add_node("source")
        for _ in range(i_count):
            self.add_node("intermediate")
        for _ in range(d_count):
            self.add_node("sink")

    def find_node_by_id(self, node_id) -> Node | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def find_nodes_by_type(self, node_type) -> list:
        return [node for node in self.nodes if node.node_type == node_type]

    def add_node(self, node_type):
        self.counters[node_type] += 1
        prefix = self.prefixes[node_type]
        node_id = f"{prefix}{self.counters[node_type]}"  # Generates S1, I1, D1, etc.
        new_node = Node(node_id, node_type)
        self.nodes.add(new_node)

    def add_edge(self, from_node_id, to_node_id):
        from_node = self.find_node_by_id(from_node_id)
        to_node = self.find_node_by_id(to_node_id)
        if from_node is not None and to_node is not None:
            from_node.add_neighbour(to_node)
            to_node.add_neighbour(from_node)
        else:
            raise ValueError("One or both nodes not found in the graph")

    def generate_random_network(self):
        pass

    def generate_one_way_network(self) -> List[str]:
        source_nodes = self.find_nodes_by_type("source")
        intermediate_nodes = self.find_nodes_by_type("intermediate")
        sink_nodes = self.find_nodes_by_type("sink")

        if len(source_nodes) != 1 or len(sink_nodes) != 1:
            raise ValueError("Graph must have exactly one source and one sink node")

        one_way: List[str] = []
        current_node = source_nodes[0]
        one_way.append(current_node.node_id)

        for next_node in intermediate_nodes:
            self.add_edge(current_node.node_id, next_node.node_id)
            current_node = next_node
            one_way.append(current_node.node_id)

        self.add_edge(current_node.node_id, sink_nodes[0].node_id)
        one_way.append(sink_nodes[0].node_id)

        return one_way

    def find_path(self, from_node_id, to_node_id) -> List[str] | None:
        def dfs(current_node, path, visited):
            if current_node.node_id == to_node_id:
                return path
            visited.add(current_node.node_id)
            for next_node in current_node.neighbours:
                if next_node.node_id not in visited:
                    result = dfs(next_node, path + [next_node.node_id], visited)
                    if result is not None:
                        return result
            return None

        start_node = self.find_node_by_id(from_node_id)
        visited = set()
        return dfs(start_node, [from_node_id], visited)

    def visualize(self, figname):
        plt.figure()

        colors = {
            'source': '#00FF004D',  # Green with 30% opacity
            'sink': '#FF00004D',  # Red with 30% opacity
            'compromised': '#FFFF004D',  # Yellow with 30% opacity
            'intermediate': '#0000FF4D'  # Blue with 30% opacity
        }

        g = nx.Graph()
        for node in self.nodes:
            g.add_node(node.node_id, label=f"{node.node_id}\n{' '.join(node.keys)}", color=colors['compromised'] if node.compromised else colors[node.node_type])
            for next_node in node.neighbours:
                if not g.has_edge(node.node_id, next_node.node_id):
                    g.add_edge(node.node_id, next_node.node_id)

        pos = nx.kamada_kawai_layout(g)
        nx.draw(g, pos,
                with_labels=True,
                labels=nx.get_node_attributes(g, 'label'),
                node_color=nx.get_node_attributes(g, 'color').values())

        plt.savefig(figname, dpi=300)
        plt.close()


class KDC:
    def __init__(self, graph, key_pool_size: int):
        self.graph = graph
        self.key_pool_size = key_pool_size
        self.key_pool = {f"k{i}" for i in range(key_pool_size)}
        self.key_pool_size_options = list(range(1, key_pool_size))

    def distribute_keys(self, key_subset_size: int):
        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        for node in self.graph.nodes:
            if node.node_type == "source":
                node.set_keys(self.key_pool.copy())
            else:
                node.set_keys(set(random.sample(sorted(self.key_pool), key_subset_size)))

    def reset_keys(self):
        for node in self.graph.nodes:
            node.empty_keys()

    def __repr__(self):
        return f"KDC({self.key_pool_size}, {self.key_pool}, {self.key_pool_size_options})"


class Attacker:
    def __init__(self, graph, kdc):
        self.graph = graph
        self.kdc = kdc
        self.key_pool_compromised: Set[str] = set()
        self.nodes_compromised: Set[Node] = set()
        self.finite_field_size = 2 ** 8

    def compromise_nodes_random(self, num_compromised_nodes: int):
        intermediate_nodes = self.graph.find_nodes_by_type("intermediate")
        if num_compromised_nodes > len(intermediate_nodes):
            raise ValueError("num_compromised_nodes cannot be larger than number of intermediate nodes")

        self.nodes_compromised = set(random.sample(intermediate_nodes, num_compromised_nodes))
        for node in self.nodes_compromised:
            node.set_compromised(True)
            self.key_pool_compromised.update(node.keys) # Add compromised node's keys if not already in key_pool_compromised

    def compromise_node_by_id(self, node_id):
        node = self.graph.find_node_by_id(node_id)
        if node is None:
            raise ValueError("Node not found in the graph")
        node.set_compromised(True)
        self.key_pool_compromised.update(node.keys)
        self.nodes_compromised.add(node)

    def get_id_of_compromised_nodes(self) -> List[str]:
        return [node.node_id for node in self.nodes_compromised]

    def reset_comromised(self):
        for node in self.graph.nodes:
            node.set_compromised(False)
        self.key_pool_compromised = set()
        self.nodes_compromised = set()

    def dpa_random_pos(self, runs: int, max_num_compromised_nodes: int):
        sink_node_id = self.graph.find_nodes_by_type("sink")[0].node_id

        # df_columns = ['compromised_nodes_id', 'path', 'is_success', 'key_pool_compromised', 'pass_paths']
        df_columns = ["compromised_nodes_id", "is_success"]

        for num_compromised_nodes in tqdm(range(1, max_num_compromised_nodes + 1), desc='Overall Progress'):
            for key_subset_size in self.kdc.key_pool_size_options:
                results = pd.DataFrame(columns=df_columns)

                for _ in tqdm(range(runs), desc=f'Running for key_subset_size={key_subset_size}'):
                    self.kdc.reset_keys()
                    self.kdc.distribute_keys(key_subset_size)
                    self.reset_comromised()
                    self.compromise_nodes_random(num_compromised_nodes)

                    # self.graph.visualize(f'./networkAnalysis/dpa_random/network_{num_compromised_nodes}c_{runs}runs_{key_subset_size}keys.png')
                    # pass_paths = []

                    paths = []
                    for node in self.nodes_compromised:
                        paths.append(self.graph.find_path(node.node_id, sink_node_id))

                    for path in paths:
                        is_success = 0  # 0 = fail, 1 = success

                        # pass_path = []

                        for n_id in path[1:]:    # Skip the compromised node
                            node = self.graph.find_node_by_id(n_id)
                            if node.keys <= self.key_pool_compromised:  # If all keys in node.keys are in key_pool_compromised
                                is_success = 1
                                # pass_path.append('success (no)')
                            else:
                                d = len(node.keys - self.key_pool_compromised)  # Number of keys not in key_pool_compromised
                                if random.randint(1, self.finite_field_size ** d) == 1:
                                # if random.randint(1, 2) == 1:
                                    is_success = 1
                                    # pass_path.append('success (yes)')
                                else:
                                    is_success = 0
                                    # pass_path.append('fail')
                                    break

                        # pass_paths.append(pass_path)

                        if is_success == 1: # If success, no need to check other compromised nodes
                            break

                    # new_row = pd.DataFrame([[self.get_id_of_compromised_nodes(), paths, is_success, self.key_pool_compromised, pass_paths]], columns=df_columns)
                    new_row = pd.DataFrame([[self.get_id_of_compromised_nodes(), is_success]], columns=df_columns)
                    results = pd.concat([results, new_row], ignore_index=True)

                results.to_csv(f'./networkAnalysis/dpa_random/results_{num_compromised_nodes}c_{runs}runs_{key_subset_size}keys.csv', index=False)

    def dpa_fix_pos(self, one_way, runs):
        intermediate_nodes_id = one_way[1:-1]
        sink_node_id = one_way[-1]

        # df_columns_debug = ["compromised_nodes_id", "key_subset_size", "is_success"]
        df_columns = ["compromised_nodes_id", "is_success"]

        for i, i_id in tqdm(enumerate(intermediate_nodes_id), desc='Overall Progress'):
            hops = len(intermediate_nodes_id) - i
            self.compromise_node_by_id(i_id)
            path_from_c_to_sink = self.graph.find_path(i_id, sink_node_id)

            for key_subset_size in range(1, self.kdc.key_pool_size + 1):

                results = pd.DataFrame(columns=df_columns)

                for _ in tqdm(range(runs), desc=f'Running for key_subset_size={key_subset_size}'):
                    self.kdc.reset_keys()
                    self.kdc.distribute_keys(key_subset_size)
                    self.reset_comromised()
                    self.compromise_node_by_id(i_id)

                    is_success = 0
                    for n_id in path_from_c_to_sink[1:]:
                        node = self.graph.find_node_by_id(n_id)
                        if node.keys <= self.key_pool_compromised:
                            is_success = 1
                        else:
                            d = len(node.keys - self.key_pool_compromised)
                            if random.randint(1, self.finite_field_size ** d) == 1:
                                is_success = 1
                            else:
                                is_success = 0
                                break

                    new_row = pd.DataFrame([[self.get_id_of_compromised_nodes(), is_success]], columns=df_columns)
                    results = pd.concat([results, new_row], ignore_index=True)

                results.to_csv(f'./networkAnalysis/dpa_fix/results_{runs}runs_{key_subset_size}keys_{hops}hops.csv', index=False)

        self.graph.visualize(f'./networkAnalysis/dpa_fix/network_{runs}runs.png')

def plot_for_dpa_random(runs: int, subset_sizes: int, num_compromised_nodes: int):
    plt.figure()
    plt.yscale('log')

    x_data = [i for i in range(1, subset_sizes + 1)]
    y_data = []
    for keys in range(1, subset_sizes + 1):
        df = pd.read_csv(f'./networkAnalysis/dpa_random/results_{num_compromised_nodes}c_{runs}runs_{keys}keys.csv')
        success_count = (df['is_success'] == 1).sum()
        success_ratio = max(success_count / runs, 1e-10)
        y_data.append(success_ratio)

    bars = plt.bar(x_data, y_data, color='red', label='Success Ratio')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, yval, va='bottom', ha='center')

    plt.title(f'{runs} runs / key_subset_size ({num_compromised_nodes} compromised nodes)')
    plt.xlabel(f'key_subset_size (key pool size = {subset_sizes + 1})')
    plt.ylabel('Logarithmic Success Ratio')
    plt.xticks(x_data)
    plt.legend()
    plt.savefig(f'./networkAnalysis/dpa_random/plot_{num_compromised_nodes}c_{runs}runs.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    runs = 100000
    key_pool_size = 10
    max_num_compromised_nodes = 5

    graph = Graph(1, 8, 1)
    kdc = KDC(graph, key_pool_size)
    attacker = Attacker(graph, kdc)

    one_way = graph.generate_one_way_network()

    hops = len(one_way) - 2
    sizes = key_pool_size - 1

    attacker.dpa_random_pos(runs, max_num_compromised_nodes)
    plot_for_dpa_random(runs, sizes, max_num_compromised_nodes)
    # attacker.dpa_fix_pos(one_way, runs)
