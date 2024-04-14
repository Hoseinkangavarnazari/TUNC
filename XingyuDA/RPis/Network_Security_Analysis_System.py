import argparse
import csv
import math
import os
import platform
import time

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Set, List, Dict, Tuple
from itertools import combinations
from tqdm import tqdm
from math import factorial


class NGraph:
    def __init__(self):
        self.type: str = None
        self.nxg = nx.Graph()
        self.picked_path: List[int] = None

    def generate_ws_graph(self, num_nodes: int):
        self.type = 'ws'

        # k represents each node's degree, it should be even and greater than 2
        # p represents the probability of rewiring. Small p will result in a graph that is more similar to a ring lattice, while large p will result in a graph that is more similar to a random graph.
        self.nxg = nx.watts_strogatz_graph(num_nodes, k=4, p=0.5)

        while not nx.is_connected(self.nxg):
            self.nxg = nx.watts_strogatz_graph(num_nodes, k=4, p=0.5)

        self.reset_node_attrs()

        nodes_list = list(self.nxg.nodes)

        source, sink = random.sample(nodes_list, 2)
        while self.nxg.has_edge(source, sink):
            source, sink = random.sample(nodes_list, 2)

        self.nxg.nodes[source]['node_type'] = 's'
        self.nxg.nodes[sink]['node_type'] = 't'

        for node in self.nxg.nodes:
            if self.nxg.nodes[node]['node_type'] is None:
                self.nxg.nodes[node]['node_type'] = 'r'

    def take_snapshot(self, filename=None):
        """
        Take a snapshot of the current graph, including reachability matrix and node attributes.
        """

        if filename is None:
            # filename = f'graph_{self.type}_{self.nxg.number_of_nodes()}nodes.csv'
            filename = f'temp_snapshot.csv'

        df_rm = nx.to_pandas_adjacency(self.nxg, dtype=int)

        attrs = {node: self.nxg.nodes[node] for node in self.nxg.nodes}
        df_attrs = pd.DataFrame.from_dict(attrs, orient='index')

        df_combined = pd.concat([df_rm, df_attrs], axis=1)   # axis=1: concat by column
        df_combined.to_csv(filename)

    def restore_snapshot(self, filename):
        """
        Restore a graph from a snapshot file.
        Only defined node attributes will be restored.
        """

        df_combined = pd.read_csv(filename, index_col=0)    # index_col=0: use the first column as index

        node_attrs_count_in_file = df_combined.shape[1] - df_combined.shape[0]  # shape[0]: number of rows, shape[1]: number of columns

        df_rm = df_combined.iloc[:, :df_combined.shape[1] - node_attrs_count_in_file]  # extract reachability matrix
        df_rm.columns = df_rm.columns.map(int)  # convert column names to int
        self.nxg = nx.from_pandas_adjacency(df_rm)

        df_attrs = df_combined.iloc[:, -node_attrs_count_in_file:]     # extract node attributes
        self.reset_node_attrs()
        defined_attrs = list(self.nxg.nodes[0].keys())

        for node in self.nxg.nodes:
            for attr in defined_attrs:
                self.nxg.nodes[node][attr] = df_attrs.loc[node, attr]

    def reset_node_attrs(self):
        for node in self.nxg.nodes:
            self.nxg.nodes[node]['node_type']: str = None           # s: source, t: sink, r: relay
            self.nxg.nodes[node]['compromised']: bool = False
            self.nxg.nodes[node]['keys']: Set[str] = None

    def visualize(self):

        s_node_id = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 's'][0]
        key_pool_size = len(self.nxg.nodes[s_node_id]['keys']) if self.nxg.nodes[s_node_id]['keys'] is not None else 0
        print(f'from visualize: key_pool_size={key_pool_size}')
        r_node_id = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') != 's'][0]   # pick one relay node at will
        keyset_size = len(self.nxg.nodes[r_node_id]['keys']) if self.nxg.nodes[r_node_id]['keys'] is not None else 0

        all_combinations = factorial(key_pool_size) // (factorial(keyset_size) * factorial(key_pool_size - keyset_size))    # nCr; factorial: 5! = 5*4*3*2*1

        used_r_keysets = set()
        for node in self.nxg.nodes:
            keys = self.nxg.nodes[node]['keys']
            used_r_keysets.add(frozenset(keys)) if keys is not None else None

        colors = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(used_r_keysets))]
        colors_map = {r_keyset: color for r_keyset, color in zip(used_r_keysets, colors)}   # zip: combine two lists into a dictionary

        node_sizes = {}
        node_labels = {}
        node_colors = [colors_map[frozenset(self.nxg.nodes[node]['keys'])] if self.nxg.nodes[node]['keys'] is not None else 'grey' for node in self.nxg.nodes]

        for node in self.nxg.nodes:
            node_type = self.nxg.nodes[node]['node_type']
            keys_info = '' if node_type == 's' else f'\n{self.nxg.nodes[node]["keys"]}'

            if node_type in ['s', 't']:
                node_sizes[node] = 1000
            elif self.nxg.nodes[node]['compromised']:
                node_sizes[node] = 800
            else:
                node_sizes[node] = 500  # Default size for other nodes

            node_labels[node] = f"{node_type[0].upper()}{node} ({len(self.nxg.nodes[node]['keys']) if self.nxg.nodes[node]['keys'] is not None else 0} keys){keys_info}"

        plt.figure(figsize=(15, 15))
        pos = nx.kamada_kawai_layout(self.nxg)
        # pos = nx.spring_layout(self.nxg, k=0.3, iterations=1000, seed=2) # k: optimal distance between nodes, iterations: number of iterations to run the spring layout algorithm, seed: seed for random state
        # pos = nx.spectral_layout(self.nxg)
        # pos = nx.circular_layout(self.nxg)

        normal_nodes = [node for node in self.nxg.nodes if not self.nxg.nodes[node]['compromised']]
        compromised_nodes = [node for node in self.nxg.nodes if self.nxg.nodes[node]['compromised']]

        nx.draw_networkx_nodes(self.nxg, pos,
                               nodelist=normal_nodes,
                               node_size=[node_sizes[n] for n in normal_nodes],
                               node_color=[node_colors[n] for n in normal_nodes])
        nx.draw_networkx_nodes(self.nxg, pos,
                               nodelist=compromised_nodes,
                               node_size=[node_sizes[n] for n in compromised_nodes],
                               node_color=[node_colors[n] for n in compromised_nodes],
                               # linewidths=1.5, edgecolors='red',
                               node_shape='*')
        nx.draw_networkx_edges(self.nxg, pos, edgelist=self.nxg.edges, edge_color='black', width=1, alpha=0.3)

        if self.picked_path is not None:
            nx.draw_networkx_edges(self.nxg, pos,
                                   edgelist=[(self.picked_path[i], self.picked_path[i + 1]) for i in range(len(self.picked_path) - 1)],
                                   edge_color='red', width=1.1)

        nx.draw_networkx_labels(self.nxg, pos, labels=node_labels, font_size=14, font_family='Open Sans')

        plt.title(f'Network with {self.nxg.number_of_nodes()} nodes\nColors Used by Non-Source Nodes={len(used_r_keysets)-1}/{all_combinations}\nKeypool Size={key_pool_size}',
                    fontsize=16,
                    fontfamily='Open Sans')

        egde_labels = {}
        for (u, v) in self.nxg.edges:
            if self.nxg.nodes[u]['node_type'] == 's' or self.nxg.nodes[v]['node_type'] == 's':
                egde_labels[(u, v)] = ''
                continue

            keyset_u_b = KDC.encode_keys(self.nxg.nodes[u]['keys'], key_pool_size)
            keyset_v_b = KDC.encode_keys(self.nxg.nodes[v]['keys'], key_pool_size)
            egde_labels[(u, v)] = KDC.hamming_distance(keyset_u_b, keyset_v_b)
        nx.draw_networkx_edge_labels(self.nxg, pos, edge_labels=egde_labels, font_size=12, font_family='Open Sans')

        plt.axis('off')
        plt.show()
        plt.close()

    def pick_path(self):
        sources = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 's']
        destinations = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 't']

        if not sources or not destinations:
            raise ValueError("Source or destination not found in the graph")

        source, destination = sources[0], destinations[0]

        # Find all simple paths from source to destination
        all_paths = list(nx.all_simple_paths(self.nxg, source=source, target=destination))

        # Determine the longest path
        longest_path = max(all_paths, key=len)
        self.picked_path = longest_path


class KDC:
    """
    Key Distribution Center
    """

    def __init__(self, graph: NGraph):
        self.graph = graph
        self.__keypool_size: int | None = None
        self.__keypool: Set[str] = set()
        self.__keysets_assigned_b: Set[str] = set()
        self.__keys_usage: Dict[str, int] = {}

    def get_keypool_size(self) -> int:
        if self.__keypool_size is None:
            raise ValueError("__keypool_size is not set yet!")
        return self.__keypool_size

    def get_keypool(self) -> Set[str]:
        if self.__keypool_size is None:
            raise ValueError("__keypool_size is not set yet!")
        return self.__keypool

    def get_keys_usage(self) -> Dict[str, int]:
        return self.__keys_usage

    def reset_distribution(self):
        for node in self.graph.nxg.nodes():
            self.graph.nxg.nodes[node]['keys'] = None

        self.__keysets_assigned_b.clear()
        self.__keys_usage.clear()

    def assign_keyset_to_r_node(self, r_node: int, keyset_b: str):
        self.graph.nxg.nodes[r_node]['keys'] = self.decode_keys(keyset_b)
        self.__keysets_assigned_b.add(keyset_b)

        for key in self.decode_keys(keyset_b):
            if key not in self.__keys_usage:
                self.__keys_usage[key] = 1
            else:
                self.__keys_usage[key] += 1

    def set_keypool_size(self, key_pool_size: int):
        self.__keypool_size = key_pool_size
        self.__keypool = {f'k{i}' for i in range(self.__keypool_size)}

    def distribute_graph_coloring_greedy(self) -> int:
        self.reset_distribution()

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 's']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        result = {node: -1 for node in subgraph.nodes}
        for node in subgraph.nodes:
            available_colors = set(range(subgraph.number_of_nodes()))
            for neighbor in subgraph.neighbors(node):
                if result[neighbor] != -1:
                    available_colors.discard(result[neighbor])
            result[node] = min(available_colors)

        num_colors = max(result.values()) + 1
        self.set_keypool_size(num_colors)

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        for node, color in result.items():
            self.assign_keyset_to_r_node(node, self.encode_keys({f'k{color}'}, num_colors))

        return num_colors


    def distribute_graph_coloring_welsh_powell(self) -> int:
        self.reset_distribution()

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 's']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        sorted_nodes = sorted(subgraph.nodes, key=lambda x: subgraph.degree[x], reverse=True)
        result = {node: -1 for node in subgraph.nodes}
        result[sorted_nodes[0]] = 0

        for i in range(1, subgraph.number_of_nodes()):
            available_colors = [True] * subgraph.number_of_nodes()
            for neighbor in subgraph.neighbors(sorted_nodes[i]):
                if result[neighbor] != -1:
                    available_colors[result[neighbor]] = False

            color = 0
            while not available_colors[color]:
                color += 1

            result[sorted_nodes[i]] = color

        num_colors = max(result.values()) + 1
        self.set_keypool_size(num_colors)

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        for node, color in result.items():
            self.assign_keyset_to_r_node(node, self.encode_keys({f'k{color}'}, num_colors))

        return num_colors

    def distribute_fix_num_keys_randomly(self, keyset_size: int, keypool_size: int):
        """
        Distributes a fixed number of keys randomly to each non-source node.
        """

        self.reset_distribution()
        self.set_keypool_size(keypool_size)

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 's']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)
        for node in subgraph.nodes:
            keyset_b = self.encode_keys(random.sample(sorted(self.get_keypool()), keyset_size), keypool_size)
            self.assign_keyset_to_r_node(node, keyset_b)

    def distribute_keys_with_CFF(self, max_c_nodes: int) -> Tuple[float, int]:

        self.reset_distribution()

        q = 10 ** (-3)
        L = math.ceil(math.e * (max_c_nodes + 1) * math.log(1 / q))    # math.e: natural number e, math.log: natural logarithm (ln)

        self.set_keypool_size(L)
        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 's']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        keyset_size = []
        for node in subgraph.nodes:
            keyset = set()
            for key in self.get_keypool():
                if random.randint(a=1, b=2 * (max_c_nodes + 1)) == 1:       # random.randint(a, b): return a random integer N such that a <= N <= b
                    keyset.add(key)

            keyset_b = self.encode_keys(keyset, L)
            self.assign_keyset_to_r_node(node, keyset_b)
            keyset_size.append(len(keyset))

        avg_keyset_size = sum(keyset_size) / len(keyset_size)
        return avg_keyset_size, L

    def distribute_fix_num_keys_with_mhd(self, keyset_size: int, keypool_size: int):
        """
        check uml diagram: dist_mh.png
        """

        self.reset_distribution()
        self.set_keypool_size(keypool_size)

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        # keysets_possible = list(combinations(self.get_keypool(), keyset_size))
        # keysets_possible_b = [self.encode_keys(keyset, keypool_size) for keyset in keysets_possible]

        # subgraph can change attributes of the original graph, but can not change layout of the original graph
        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 's']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        def pick_next_node_to_assign(g: nx.Graph, group: Set[int]) -> int | None:
            """
            The input is a set of nodes and the output is one of the nodes with the largest degree among the neighbours of that set of nodes.
            """

            all_neighbors = set()

            for node in group:
                all_neighbors.update(g.neighbors(node))

            all_neighbors = all_neighbors - group

            if all_neighbors:
                neighbor = max(all_neighbors, key=lambda x: g.degree[x])
                return neighbor

            return None

        def find_keysets_with_mhds(g: nx.Graph, node: int, keysets_possible_b: List[str], keypool_size) -> List[str]:
            """
            In keysets_possible_b, find all subsets that have the maximum hamming distance sum to all neighbours.
            """
            max_hd_sum = -1
            keysets_with_mhd_b = []

            for keyset_b in keysets_possible_b:
                hd_sum = 0

                for neighbor in g.neighbors(node):
                    if g.nodes[neighbor]['keys'] is not None:
                        hd_sum += self.hamming_distance(keyset_b, self.encode_keys(g.nodes[neighbor]['keys'], keypool_size))

                if hd_sum > max_hd_sum:
                    max_hd_sum = hd_sum
                    keysets_with_mhd_b = [keyset_b]
                elif hd_sum == max_hd_sum:
                    keysets_with_mhd_b.append(keyset_b)

            return keysets_with_mhd_b

        def pick_best_keyset(g: nx.Graph, node: int, keypool_size, keyset_size_t) -> str:

            if not self.__keysets_assigned_b:
                keyset_best_b = ['1'] * keyset_size_t + ['0'] * (keypool_size - keyset_size_t)
                random.shuffle(keyset_best_b)
                return ''.join(keyset_best_b)
            elif len(self.__keysets_assigned_b) == 1:
                keyset_best_b = ['0'] * keypool_size
                one_placed = 0
                for i in range(keypool_size):
                    if one_placed < keyset_size_t and list(self.__keysets_assigned_b)[0][i] == '0':
                        keyset_best_b[i] = '1'
                        one_placed += 1
                return ''.join(keyset_best_b)
            else:
                list_keys_used = list(self.__keysets_assigned_b)
                A = list_keys_used[0]
                B = list_keys_used[1]
                keyset_best_b = ['0'] * keypool_size
                for i in range(keypool_size):
                    if A[i] == B[i]:
                        keyset_best_b[i] = '1' if A[i] == '0' else '0'

                one_placed =keyset_best_b.count('1')
                if one_placed < keyset_size_t:
                    for i in range(keypool_size):
                        if one_placed >= keyset_size_t:
                            break
                        if keyset_best_b[i] == '0' and A[i] != B[i]:
                            keyset_best_b[i] = '1'
                            one_placed += 1
                elif one_placed > keyset_size_t:
                    index_ones = [i for i, bit in enumerate(keyset_best_b) if bit == '1']
                    index_to_flip = random.sample(index_ones, one_placed - keyset_size_t)
                    for i in index_to_flip:
                        keyset_best_b[i] = '0'

                return ''.join(keyset_best_b)



        assigned_group = set()
        while len(assigned_group) < len(subgraph.nodes):
            if not assigned_group:
                node = max(non_source_nodes, key=lambda x: subgraph.degree[x])
            else:
                node = pick_next_node_to_assign(subgraph, assigned_group)

            keyset_chosen_b = pick_best_keyset(subgraph, node, keypool_size, keyset_size)

            self.assign_keyset_to_r_node(node, keyset_chosen_b)
            assigned_group.add(node)

    @staticmethod
    def encode_keys(node_keys, keypool_size):
        return ''.join(['1' if f'k{i}' in node_keys else '0' for i in range(keypool_size)])

    def decode_keys(self, encoded_keys):
        return {f'k{i}' for i in range(self.get_keypool_size()) if encoded_keys[i] == '1'}

    @staticmethod
    def hamming_distance(str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))


class Attacker:
    def __init__(self, graph: NGraph, kdc: KDC):
        self.graph = graph
        self.kdc = kdc
        self.keypool_compromised: Set[str] = set()
        self.finite_field_size = 2 ** 2

    def compromise_node_randomly_on_path(self, num_compromised_nodes: int) -> List[int]:
        """
        Compromises a fixed number of nodes randomly along the path.
        """

        if num_compromised_nodes > len(self.graph.picked_path) - 2:
            raise ValueError("num_compromised_nodes cannot be larger than number of relay nodes in the path")

        # only compromise intermediate nodes
        compromised_nodes = random.sample(self.graph.picked_path[1:-1], num_compromised_nodes)

        for node in compromised_nodes:
            self.graph.nxg.nodes[node]['compromised'] = True
            if self.graph.nxg.nodes[node]['keys'] is not None:
                self.keypool_compromised.update(self.graph.nxg.nodes[node]['keys'])

        return compromised_nodes

    # def compromise_node(self, node: int):
    #     """
    #     Compromises a specific node.
    #     """
    #
    #     if node not in self.graph.nxg.nodes:
    #         raise ValueError("Node not found in the graph")
    #     elif self.graph.nxg.nodes[node]['node_type'] != 'r':
    #         raise ValueError("Node is not a relay node")
    #
    #     self.graph.nxg.nodes[node]['compromised'] = True
    #     self.keypool_compromised.update(self.graph.nxg.nodes[node]['keys'])

    def reset_comromise(self):
        for node in self.graph.nxg.nodes():
            self.graph.nxg.nodes[node]['compromised'] = False
        self.keypool_compromised.clear()

    def run_single_attack(self, attack_type: str, distr_strategy: str, attack_model: str, keyset_size: int | None, keypool_size: int | None, num_compromised_nodes: int):
        if distr_strategy == 'rd':
            actual_avg_keyset_size = keyset_size
            actual_keypool_size = keypool_size
            self.kdc.distribute_fix_num_keys_randomly(keyset_size, keypool_size)
        elif distr_strategy == 'mhd':
            actual_avg_keyset_size = keyset_size
            actual_keypool_size = keypool_size
            self.kdc.distribute_fix_num_keys_with_mhd(keyset_size, keypool_size)
        elif distr_strategy == 'cff':
            actual_avg_keyset_size, actual_keypool_size = self.kdc.distribute_keys_with_CFF(max_c_nodes=num_compromised_nodes)
        elif distr_strategy == 'greedy':
            actual_avg_keyset_size = 1
            actual_keypool_size = self.kdc.distribute_graph_coloring_greedy()
        elif distr_strategy == 'wp':
            actual_avg_keyset_size = 1
            actual_keypool_size = self.kdc.distribute_graph_coloring_welsh_powell()
        else:
            raise ValueError(f"Unknown distribution strategy: {distr_strategy}")

        self.reset_comromise()
        compromised_nodes = self.compromise_node_randomly_on_path(num_compromised_nodes)

        count_checks = 0        # number of checks performed by non-compromised nodes
        first_c_node_index = next((i for i, node in enumerate(self.graph.picked_path) if self.graph.nxg.nodes[node]['compromised']), None)
        if first_c_node_index is not None:
            for node in self.graph.picked_path[first_c_node_index:]:
                if not self.graph.nxg.nodes[node]['compromised']:
                    count_checks += 1

        is_success = None

        if attack_type == 'dpa':
            if attack_model == 'other':
                """
                check uml diagram: other_dpa.png
                """

                is_success = 0  # assume the attack fails

                for c_node in compromised_nodes:
                    path_from_c = self.graph.picked_path[self.graph.picked_path.index(c_node):]
                    path_success = 1  # assume the path succeeds

                    for nd in path_from_c[1:]:  # Skip compromised node
                        d = len(self.graph.nxg.nodes[nd]['keys'] - self.keypool_compromised)
                        if d == 0:
                            continue
                        elif random.randint(1, self.finite_field_size ** d) != 1:
                            path_success = 0
                            break

                    if path_success == 1:
                        is_success = 1  # If at least one path succeeds, the entire attack succeeds
                        break

            elif attack_model == 'our':
                """
                check uml diagram: our_dpa.png
                """

                is_success = 1      # assume the attack succeeds
                latest_compromised_node = next((node for node in self.graph.picked_path if self.graph.nxg.nodes[node]['compromised']), None)

                for node in self.graph.picked_path[self.graph.picked_path.index(latest_compromised_node):]:
                    if self.graph.nxg.nodes[node]['compromised']:
                        latest_compromised_node = node
                    else:
                        # TODO: What if there are MACs left that were modified by the last compromised node?
                        d = len(self.graph.nxg.nodes[node]['keys'] - self.graph.nxg.nodes[latest_compromised_node]['keys'])
                        if random.randint(1, self.finite_field_size ** d) != 1:
                            is_success = 0
                            break

        elif attack_type == 'tpa':
            if attack_model == 'our':
                """
                check uml diagram: our_tpa.png
                """

                is_success = 1
                compromised_key = None

                first_c_node_index = next((i for i, node in enumerate(self.graph.picked_path) if self.graph.nxg.nodes[node]['compromised']), None)

                for node in self.graph.picked_path[first_c_node_index:]:
                    if self.graph.nxg.nodes[node]['compromised']:
                        compromised_key = random.choice(list(self.kdc.get_keypool()))
                    elif compromised_key in self.graph.nxg.nodes[node]['keys']:
                        is_success = 0
                        break

            elif attack_model == 'other':
                """
                check uml diagram: other_tpa.png
                """

                is_success = 0

                for c_node in compromised_nodes:
                    path_from_c = self.graph.picked_path[self.graph.picked_path.index(c_node):]
                    path_success = 1
                    compromised_key = random.choice(list(self.kdc.get_keypool()))

                    for nd in path_from_c[1:]:  # Skip compromised node
                        if not self.graph.nxg.nodes[nd]['compromised']:
                            if compromised_key in self.graph.nxg.nodes[nd]['keys']:
                                path_success = 0
                                break

                    if path_success == 1:
                        is_success = 1
                        break

        return compromised_nodes, is_success, count_checks, actual_avg_keyset_size, actual_keypool_size


def measure_time(method, *args, **kwargs):
    start = time.time()
    method(*args, **kwargs)
    end = time.time()
    return end - start


def compare_time():
    total_runs = 1000
    file = 'times.csv'
    methods = ['Greedy MKD', 'Welsh-Powell MKD', 'CFF-based KD', 'Max-Hamming PKD\n(Config 1)', 'Max-Hamming PKD\n(Config 2)']

    time_greedy = 0
    time_wp = 0
    time_cff = 0
    time_mhd = 0
    time_mhd_2 = 0

    if os.path.exists(file):
        data = pd.read_csv(file)
        avg_time_greedy = data[data['Method'] == 'Greedy MKA']['Time (ms)'].values[0]
        avg_time_wp = data[data['Method'] == 'Welsh-Powell MKA']['Time (ms)'].values[0]
        avg_time_cff = data[data['Method'] == 'CFF-based']['Time (ms)'].values[0]
        avg_time_mhd = data[data['Method'] == 'Max-Hamming PKA\n(Config 1)']['Time (ms)'].values[0]
        avg_time_mhd_2 = data[data['Method'] == 'Max-Hamming PKA\n(Config 2)']['Time (ms)'].values[0]

    else:
        g1 = NGraph()
        g1.restore_snapshot(filename='/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/Codes/networkAnalysis/networks/10n.csv')
        g1.pick_path()

        kdc = KDC(graph=g1)

        for _ in range(total_runs):
            time_greedy += measure_time(kdc.distribute_graph_coloring_greedy)
            time_wp += measure_time(kdc.distribute_graph_coloring_welsh_powell)
            time_cff += measure_time(kdc.distribute_keys_with_CFF, max_c_nodes=4)
            time_mhd += measure_time(kdc.distribute_fix_num_keys_with_mhd, keyset_size=5, keypool_size=10)
            time_mhd_2 += measure_time(kdc.distribute_fix_num_keys_with_mhd, keyset_size=5, keypool_size=11)

        # Calculate average times
        avg_time_greedy = (time_greedy / total_runs) * 1000
        avg_time_wp = (time_wp / total_runs) * 1000
        avg_time_cff = (time_cff / total_runs) * 1000
        avg_time_mhd = (time_mhd / total_runs) * 1000
        avg_time_mhd_2 = (time_mhd_2 / total_runs) * 1000

        times = [avg_time_greedy, avg_time_wp, avg_time_cff, avg_time_mhd, avg_time_mhd_2]

        with open('times.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Time (ms)'])
            for i in range(len(methods)):
                writer.writerow([methods[i], times[i]])

    times = [avg_time_greedy, avg_time_wp, avg_time_cff, avg_time_mhd, avg_time_mhd_2]
    color_map = {
        'Greedy MKD': '#7D2E8D',  # 紫色
        'Welsh-Powell MKD': '#76AB2F',  # 绿色
        'CFF-based KD': '#0071BC',  # 蓝色
        'Max-Hamming PKD\n(Config 1)': '#D85218',  # 橙色
        'Max-Hamming PKD\n(Config 2)': '#ECB01F',  # 黄色
    }

    plt.figure(figsize=(10, 6))
    plt.bar(methods, times, width=0.3, color=[color_map[method] for method in methods])
    plt.xlabel('Key Distribution Mechanism', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.yscale('log')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.show()


def calculate_safekeys(kd):
    g1 = NGraph()

    if platform.system() == 'Darwin':   # Mac OS
        g1.restore_snapshot(filename='/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/Codes/RPis/4n.csv')
    else:
        g1.restore_snapshot(filename='/home/xingyu/NoteOnGithub/Diplomarbeit/Codes/RPis/4n.csv')

    kdc = KDC(graph=g1)

    if kd == 'mhd925':
        kdc.distribute_fix_num_keys_with_mhd(keyset_size=9, keypool_size=25)
    elif kd == 'mhd1020':
        kdc.distribute_fix_num_keys_with_mhd(keyset_size=10, keypool_size=20)
    elif kd == 'cff':
        kdc.distribute_keys_with_CFF(max_c_nodes=1)

    # caulculate the number of keys that hold by node 2, but not hold by node 1
    d_21 = len(g1.nxg.nodes[2]['keys'] - g1.nxg.nodes[1]['keys'])
    d_31 = len(g1.nxg.nodes[3]['keys'] - g1.nxg.nodes[1]['keys'])
    # g1.visualize()
    return d_21, d_31


def run_simulation(batch, total_batches):
    g1 = NGraph()

    if platform.system() == 'Darwin':   # Mac OS
        g1.restore_snapshot(filename='/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/Codes/networkAnalysis/networks/10n.csv')
        m_result_folder = '/Users/xingyuzhou/Downloads/dpa_mhd_our_106_22_all'
    else:
        g1.restore_snapshot(filename='/home/xingyu/NoteOnGithub/Diplomarbeit/Codes/networkAnalysis/networks/10n.csv')
        m_result_folder = '/home/xingyu/Downloads/dpa_mhd_our_5_106_22'

    g1.pick_path()

    m_keypool_size = 10
    m_total_runs = 10 ** 6
    m_attack_type = 'dpa'       # tpa | dpa
    m_distr_strategy = 'mhd'  # rd | mhd | cff | greedy | wp
    m_attack_model = 'our'      # our | other

    kdc = KDC(graph=g1)
    attacker = Attacker(graph=g1, kdc=kdc)

    max_num_compromised_nodes = g1.nxg.number_of_nodes() - 2

    os.makedirs(m_result_folder, exist_ok=True)

    if batch == total_batches:
        m_runs = m_total_runs - (total_batches - 1) * (m_total_runs // total_batches)
    else:
        m_runs = m_total_runs // total_batches

    for num_compromised_nodes in range(1, max_num_compromised_nodes + 1, 1):
    # for num_compromised_nodes in [2]:

        if m_distr_strategy == 'cff':
            results = []
            file_name = f'{m_result_folder}/csv{batch}_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c.csv'
            if os.path.exists(file_name):
                print(f'File {file_name} already exists, skipping...')
                continue

            for _ in tqdm(range(m_runs), desc=f'Running {m_attack_type}_{m_distr_strategy}_{m_attack_model} (batch={batch}, c={num_compromised_nodes}/{max_num_compromised_nodes})'):
                compromised_nodes, is_success, checks, actual_avg_keyset_size, actual_keypool_size = attacker.run_single_attack(attack_type=m_attack_type,
                                                                                   distr_strategy=m_distr_strategy,
                                                                                   attack_model=m_attack_model,
                                                                                   keyset_size=None,
                                                                                   keypool_size=None,
                                                                                   num_compromised_nodes=num_compromised_nodes)
                results.append([compromised_nodes, is_success, checks, actual_avg_keyset_size, actual_keypool_size])

            result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success', 'count_checks', 'avg_keyset_size', 'keypool_size'])
            result.to_csv(file_name, index=False)

        elif m_distr_strategy in ['greedy', 'wp']:
            results = []
            file_name = f'{m_result_folder}/csv{batch}_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c.csv'
            if os.path.exists(file_name):
                print(f'File {file_name} already exists, skipping...')
                continue

            for _ in tqdm(range(m_runs), desc=f'Running {m_attack_type}_{m_distr_strategy}_{m_attack_model} (batch={batch}, c={num_compromised_nodes}/{max_num_compromised_nodes})'):
                compromised_nodes, is_success, checks, actual_avg_keyset_size, actual_keypool_size = attacker.run_single_attack(attack_type=m_attack_type,
                                                                                   distr_strategy=m_distr_strategy,
                                                                                   attack_model=m_attack_model,
                                                                                   keyset_size=None,
                                                                                   keypool_size=None,
                                                                                   num_compromised_nodes=num_compromised_nodes)
                results.append([compromised_nodes, is_success, checks, actual_avg_keyset_size, actual_keypool_size])

            result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success', 'count_checks', 'avg_keyset_size', 'keypool_size'])
            result.to_csv(file_name, index=False)

        else:
            for keyset_size in range(1, m_keypool_size, 1):
            # for keyset_size in [5]:
                results = []
                file_name = f'{m_result_folder}/csv{batch}_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c_{keyset_size}keys.csv'

                if os.path.exists(file_name):
                    print(f'File {file_name} already exists, skipping...')
                    continue

                for _ in tqdm(range(m_runs), desc=f'Running {m_attack_type}_{m_distr_strategy}_{m_attack_model} (batch={batch}, c={num_compromised_nodes}/{max_num_compromised_nodes}, keys={keyset_size}/{m_keypool_size})'):
                    compromised_nodes, is_success, checks, actual_avg_keyset_size, actual_keypool_size = attacker.run_single_attack(attack_type=m_attack_type,
                                                                                       distr_strategy=m_distr_strategy,
                                                                                       attack_model=m_attack_model,
                                                                                       keyset_size=keyset_size,
                                                                                       keypool_size=m_keypool_size,
                                                                                       num_compromised_nodes=num_compromised_nodes)
                    results.append([compromised_nodes, is_success, checks, actual_avg_keyset_size, actual_keypool_size])

                result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success', 'count_checks', 'avg_keyset_size', 'keypool_size'])
                result.to_csv(file_name, index=False)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Run simulation with specified batch number')
    # parser.add_argument('batch', type=int, help='Current batch number for the simulation')
    # parser.add_argument('total_batches', type=int, help='Total number of batches for the simulation')
    # args = parser.parse_args()
    # run_simulation(batch=args.batch, total_batches=args.total_batches)  # use start.sh

    # 运行 1000 次calculate_safekeys， 统计 d_21 和 d_31 的平均值
    d_21_cff_sum = 0
    d_31_cff_sum = 0
    for i in range(1000):
        d_21, d_31 = calculate_safekeys('cff')
        d_21_cff_sum += d_21
        d_31_cff_sum += d_31

    d_21_cff_avg = d_21_cff_sum / 1000
    d_31_cff_avg = d_31_cff_sum / 1000
    print(f'Average d_21: {d_21_cff_avg}, Average d_31: {d_31_cff_avg}')


    d_21_mhd_sum = 0
    d_31_mhd_sum = 0
    for i in range(1000):
        d_21, d_31 = calculate_safekeys('mhd')
        d_21_mhd_sum += d_21
        d_31_mhd_sum += d_31

    d_21_mhd_avg = d_21_mhd_sum / 1000
    d_31_mhd_avg = d_31_mhd_sum / 1000
    print(f'Average d_21: {d_21_mhd_avg}, Average d_31: {d_31_mhd_avg}')
