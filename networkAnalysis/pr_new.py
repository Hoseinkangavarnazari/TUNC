import argparse
import math
import os
import platform

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
            filename = f'graph_temp.csv'

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

            node_labels[node] = f"{node_type[0].upper()}({node}), ({len(self.nxg.nodes[node]['keys']) if self.nxg.nodes[node]['keys'] is not None else 0} keys){keys_info}"

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

        nx.draw_networkx_labels(self.nxg, pos, labels=node_labels)

        plt.title(f'Network with {self.nxg.number_of_nodes()} nodes\nColors Used by Non-Source Nodes={len(used_r_keysets)-1}/{all_combinations}\nKeypool Size={key_pool_size}')
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
            keyset_b = self.encode_keys(random.sample(sorted(self.get_keypool()), keyset_size))
            self.assign_keyset_to_r_node(node, keyset_b)

    def distribute_keys_with_CFF(self, max_c_nodes: int):

        self.reset_distribution()

        # use setup in dual-HMAC
        q = 10 ** (-3)  # ten to the power of -3
        Pr = 1 / (2 * (max_c_nodes + 1))
        L = math.ceil(math.e * (max_c_nodes + 1) * math.log(1 / q))    # math.e: natural number e, math.log: natural logarithm (ln)
        avg_keyset_size = math.ceil((math.e / 2) * math.log(1 / q))

        self.set_keypool_size(L)
        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        # TODO: If sink nodes can't be compromised, why don't we assign all the keys to the sink nodes?
        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 's']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)
        for node in subgraph.nodes:
            keyset = set()
            for key in self.get_keypool():
                if random.randint(a=1, b=2 * (max_c_nodes + 1)) == 1:       # random.randint(a, b): return a random integer N such that a <= N <= b
                    keyset.add(key)

            keyset_b = self.encode_keys(keyset)
            self.assign_keyset_to_r_node(node, keyset_b)

    def distribute_fix_num_keys_with_mhd(self, keyset_size: int, keypool_size: int):
        """
        check uml diagram: dist_mh.png
        """

        self.reset_distribution()
        self.set_keypool_size(keypool_size)

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 's']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.get_keypool().copy()

        # TODO: when keypool_size is large, the number of possible keysets is too large to be computed
        keysets_possible = list(combinations(self.get_keypool(), keyset_size))
        keysets_possible_b = [self.encode_keys(keyset) for keyset in keysets_possible]

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

        def find_keysets_with_mhds(g: nx.Graph, node: int, keysets_possible_b: List[str]) -> List[str]:
            """
            In keysets_possible_b, find all subsets that have the maximum hamming distance sum to all neighbours.
            """
            max_hd_sum = -1
            keysets_with_mhd_b = []

            for keyset_b in keysets_possible_b:
                hd_sum = 0

                for neighbor in g.neighbors(node):
                    if g.nodes[neighbor]['keys'] is not None:
                        hd_sum += self.hamming_distance(keyset_b, self.encode_keys(g.nodes[neighbor]['keys']))

                if hd_sum > max_hd_sum:
                    max_hd_sum = hd_sum
                    keysets_with_mhd_b = [keyset_b]
                elif hd_sum == max_hd_sum:
                    keysets_with_mhd_b.append(keyset_b)

            return keysets_with_mhd_b

        def pick_best_keyset(g: nx.Graph, node: int, keysets_possible_b: List[str]) -> str:
            """
            1. for the case where node keys is None, treat the distance to the Hamming as 0.
            2. call find_subsets_with_max_hd_sum() to get possible subsets.
            3. among the possible subsets, select the unused subsets as the new possible subsets, if all of them have been used, select a subset randomly and return it.
            4. choose a random subset among the new possible subsets and return it.
            """

            keysets_with_mhd_b = find_keysets_with_mhds(g, node, keysets_possible_b)
            keysets_unused_b = [keyset for keyset in keysets_with_mhd_b if keyset not in self.__keysets_assigned_b]

            total_keys_needed = len(subgraph.nodes) * keyset_size
            key_usage_freq = {key: self.__keys_usage.get(key, 0) / total_keys_needed for key in self.get_keypool()}
            keysets_to_consider_b = keysets_unused_b if keysets_unused_b else keysets_with_mhd_b

            # keyset_best_b = None
            # min_deviation = float('inf')
            # for keyset_b in keysets_to_consider_b:
            #     deviation = sum(abs(key_usage_freq[key] - 1 / keypool_size) for key in self.decode_keys(keyset_b))
            #     if deviation < min_deviation:
            #         min_deviation = deviation
            #         keyset_best_b = keyset_b

            keyset_best_b = random.choice(keysets_to_consider_b)

            return keyset_best_b

        assigned_group = set()
        while len(assigned_group) < len(subgraph.nodes):
            if not assigned_group:
                node = max(non_source_nodes, key=lambda x: subgraph.degree[x])
            else:
                node = pick_next_node_to_assign(subgraph, assigned_group)

            keyset_chosen_b = pick_best_keyset(subgraph, node, keysets_possible_b)

            self.assign_keyset_to_r_node(node, keyset_chosen_b)
            assigned_group.add(node)

    def encode_keys(self, node_keys):
        return ''.join(['1' if f'k{i}' in node_keys else '0' for i in range(self.get_keypool_size())])

    def decode_keys(self, encoded_keys):
        return {f'k{i}' for i in range(self.get_keypool_size()) if encoded_keys[i] == '1'}

    def hamming_distance(self, str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))


class Attacker:
    def __init__(self, graph: NGraph, kdc: KDC):
        self.graph = graph
        self.kdc = kdc
        self.keypool_compromised: Set[str] = set()
        self.finite_field_size = 2 ** 8

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

    def run_single_attack(self, attack_type: str, distr_strategy: str, attack_model: str, keyset_size: int | None, keypool_size: int | None, num_compromised_nodes: int) -> Tuple[List[int], int, int]:
        if distr_strategy == 'rd':
            self.kdc.distribute_fix_num_keys_randomly(keyset_size, keypool_size)
        elif distr_strategy == 'mhd':
            self.kdc.distribute_fix_num_keys_with_mhd(keyset_size, keypool_size)
        elif distr_strategy == 'cff':
            self.kdc.distribute_keys_with_CFF(max_c_nodes=num_compromised_nodes)
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
                        # TODO: Can I make possibility 10 times higher by let it > 100?
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

        return compromised_nodes, is_success, count_checks


def demo():
    g1 = NGraph()
    # g1.generate_ws_graph(num_nodes=12)
    g1.restore_snapshot(filename='/Users/xingyuzhou/TUNC/networkAnalysis/networks/12n.csv')
    g1.pick_path()

    kdc = KDC(graph=g1)

    attacker = Attacker(graph=g1, kdc=kdc)
    compromised_nodes, is_success, count_checks = attacker.run_single_attack(attack_type='dpa',
                                                                                distr_strategy='cff',
                                                                                attack_model='our',
                                                                                keyset_size=None,
                                                                                keypool_size=None,
                                                                                num_compromised_nodes=3)
    print(f'compromised_nodes: {compromised_nodes}, is_success: {is_success}, count_checks: {count_checks}')
    g1.visualize()
    # compromised_nodes, is_success, count_checks = attacker.run_single_attack(attack_type='dpa',
    #                                                                          distr_strategy='mhd',
    #                                                                          attack_model='our',
    #                                                                          keyset_size=10,
    #                                                                          keypool_size=20,
    #                                                                          num_compromised_nodes=3)
    # print(f'compromised_nodes: {compromised_nodes}, is_success: {is_success}, count_checks: {count_checks}')

    # g1.take_snapshot(filename='/Users/xingyuzhou/Downloads/new.csv')


def run_simulation(batch, total_batches):
    g1 = NGraph()

    if platform.system() == 'Darwin':
        g1.restore_snapshot(filename='/Users/xingyuzhou/TUNC/networkAnalysis/networks/12n.csv')
        m_result_folder = '/Users/xingyuzhou/Downloads/test'
    else:
        g1.restore_snapshot(filename='/home/xingyu/Downloads/12n.csv')
        m_result_folder = '/home/xingyu/Downloads/mhd10w'

    g1.pick_path()

    m_key_pool_size = 12
    m_total_runs = 10
    m_attack_type = 'dpa'       # tpa | dpa
    m_distr_strategy = 'mhd'  # rd | mhd | cff
    m_attack_model = 'other'      # our | other

    kdc = KDC(graph=g1)
    attacker = Attacker(graph=g1, kdc=kdc)

    max_num_compromised_nodes = g1.nxg.number_of_nodes() - 2
    compromised_node_options = [i for i in range(1, max_num_compromised_nodes + 1)]
    keyset_size_options = [i for i in range(1, m_key_pool_size)]

    os.makedirs(m_result_folder, exist_ok=True)

    if batch == total_batches:
        m_runs = m_total_runs - (total_batches - 1) * (m_total_runs // total_batches)
    else:
        m_runs = m_total_runs // total_batches

    for num_compromised_nodes in compromised_node_options:
    # for num_compromised_nodes in [8]:

        if m_distr_strategy == 'cff':
            results = []
            file_name = f'{m_result_folder}/csv{batch}_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c.csv'
            if os.path.exists(file_name):
                print(f'File {file_name} already exists, skipping...')
                continue

            for _ in tqdm(range(m_runs), desc=f'Running {m_attack_type} ({m_distr_strategy}, {m_attack_model}) for c=({num_compromised_nodes}/{max_num_compromised_nodes})'):
                compromised_nodes, is_success, checks = attacker.run_single_attack(attack_type=m_attack_type,
                                                                                   distr_strategy=m_distr_strategy,
                                                                                   attack_model=m_attack_model,
                                                                                   keyset_size=None,
                                                                                   keypool_size=None,
                                                                                   num_compromised_nodes=num_compromised_nodes)
                results.append([compromised_nodes, is_success, checks])

            result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success', 'count_checks'])
            result.to_csv(file_name, index=False)

        else:
            for keyset_size in keyset_size_options:
            # for keyset_size in [3]:
                results = []
                file_name = f'{m_result_folder}/csv{batch}_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c_{keyset_size}keys.csv'

                if os.path.exists(file_name):
                    print(f'File {file_name} already exists, skipping...')
                    continue

                for _ in tqdm(range(m_runs), desc=f'Running {m_attack_type} ({m_distr_strategy}, {m_attack_model}) for c=({num_compromised_nodes}/{max_num_compromised_nodes}), keys=({keyset_size}/{len(keyset_size_options)})'):
                    compromised_nodes, is_success, checks = attacker.run_single_attack(attack_type=m_attack_type,
                                                                                       distr_strategy=m_distr_strategy,
                                                                                       attack_model=m_attack_model,
                                                                                       keyset_size=keyset_size,
                                                                                       keypool_size=m_key_pool_size,
                                                                                       num_compromised_nodes=num_compromised_nodes)
                    results.append([compromised_nodes, is_success, checks])

                result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success', 'count_checks'])
                result.to_csv(file_name, index=False)


if __name__ == "__main__":
    # demo()

    parser = argparse.ArgumentParser(description='Run simulation with specified batch number')
    parser.add_argument('batch', type=int, help='Current batch number for the simulation')
    parser.add_argument('total_batches', type=int, help='Total number of batches for the simulation')
    args = parser.parse_args()
    run_simulation(batch=args.batch, total_batches=args.total_batches)  # for example: python3 pr_new.py 1 10
