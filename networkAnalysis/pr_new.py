import argparse
import concurrent.futures
import os
import re

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

        self.nxg.nodes[source]['node_type'] = 'source'
        self.nxg.nodes[sink]['node_type'] = 'destination'

        for node in self.nxg.nodes:
            if self.nxg.nodes[node]['node_type'] is None:
                self.nxg.nodes[node]['node_type'] = 'intermediate'

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
            self.nxg.nodes[node]['node_type']: str = None
            self.nxg.nodes[node]['compromised']: bool = False
            self.nxg.nodes[node]['keys']: Set[str] = None

    def visualize(self):

        source = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 'source'][0]
        key_pool_size = len(self.nxg.nodes[source]['keys']) if self.nxg.nodes[source]['keys'] is not None else 0
        non_source = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') != 'source'][0]
        subset_size = len(self.nxg.nodes[non_source]['keys']) if self.nxg.nodes[non_source]['keys'] is not None else 0
        all_combinations = factorial(key_pool_size) // (factorial(subset_size) * factorial(key_pool_size - subset_size))    # nCr; factorial: 5! = 5*4*3*2*1

        used_key_subsets = set()
        for node in self.nxg.nodes:
            keys = self.nxg.nodes[node]['keys']
            used_key_subsets.add(frozenset(keys)) if keys is not None else None

        colors = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(used_key_subsets))]
        colors_map = {subset: color for subset, color in zip(used_key_subsets, colors)}   # zip: combine two lists into a dictionary

        node_sizes = {}
        node_labels = {}
        node_colors = [colors_map[frozenset(self.nxg.nodes[node]['keys'])] if self.nxg.nodes[node]['keys'] is not None else 'grey' for node in self.nxg.nodes]

        for node in self.nxg.nodes:
            node_type = self.nxg.nodes[node]['node_type']

            if node_type in ['source', 'destination']:
                node_sizes[node] = 1000
            elif self.nxg.nodes[node]['compromised']:
                node_sizes[node] = 800
            else:
                node_sizes[node] = 500  # Default size for other nodes

            node_labels[node] = f"{node_type[0].upper()}-{node}\n{self.nxg.nodes[node]['keys']}"

        plt.figure(figsize=(12, 12))
        pos = nx.kamada_kawai_layout(self.nxg)
        # pos = nx.spring_layout(self.nxg, k=0.3, iterations=1000, seed=2) # k: optimal distance between nodes, iterations: number of iterations to run the spring layout algorithm, seed: seed for random state
        # pos = nx.spectral_layout(self.nxg)
        # pos = nx.circular_layout(self.nxg)

        compromised_nodes = [node for node in self.nxg.nodes if self.nxg.nodes[node]['compromised']]
        normal_nodes = [node for node in self.nxg.nodes if not self.nxg.nodes[node]['compromised']]

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

        plt.title(f'Network with {self.nxg.number_of_nodes() - 1} non-source nodes\nColors Used={len(used_key_subsets)-1}/{all_combinations} (source excluded), Key Pool Size={key_pool_size}')
        plt.axis('off')
        plt.show()
        plt.close()

    def pick_path(self):
        sources = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 'source']
        destinations = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 'destination']

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

    def __init__(self, graph: NGraph, key_pool_size: int):
        self.graph = graph
        self.key_pool_size = key_pool_size
        self.key_pool: Set[str] = {f"k{i}" for i in range(key_pool_size)}
        self.key_usage: Dict[str, int] = {f"k{i}": 0 for i in range(key_pool_size)}
        self.used_key_subsets: Set[str] = set()

    def distribute_fix_num_keys_randomly(self, key_subset_size: int):
        """
        Distributes a fixed number of keys randomly to each non-source node.
        """

        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 'source']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.key_pool.copy()

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 'source']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)
        for node in subgraph.nodes:
            subgraph.nodes[node]['keys'] = set(random.sample(sorted(self.key_pool), key_subset_size))

    def reset_keys(self):
        for node in self.graph.nxg.nodes():
            self.graph.nxg.nodes[node]['keys'] = None

    def encode_keys(self, node_keys):
        return ''.join(['1' if f'k{i}' in node_keys else '0' for i in range(self.key_pool_size)])

    def decode_keys(self, encoded_keys):
        return {f'k{i}' for i in range(self.key_pool_size) if encoded_keys[i] == '1'}

    def hamming_distance(self, str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def distribute_fix_num_keys_with_max_hamming_distance(self, key_subset_size: int, strategy='nf'):
        """
        check uml diagram: dist_mh.png
        """

        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] == 'source']
        for node in source_nodes:
            self.graph.nxg.nodes[node]['keys'] = self.key_pool.copy()

        possible_subsets = list(combinations(self.key_pool, key_subset_size))
        possible_subsets_encoded = [self.encode_keys(key_subset) for key_subset in possible_subsets]

        # subgraph can change attributes of the original graph, but can not change layout of the original graph
        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 'source']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        def find_neighbor_of_group_with_max_degree(g: nx.Graph, group: Set[int]) -> int:
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

        def find_subsets_with_max_hd_sum(g: nx.Graph, node: int, possible_subsets_encoded: List[str]) -> List[str]:
            """
            In possible_subsets_encoded, find all subsets that have the maximum hamming distance sum to all neighbours.
            """
            max_hd_sum = -1
            subsets_with_max_hd = []

            for subset in possible_subsets_encoded:
                hd_sum = 0

                for neighbor in g.neighbors(node):
                    if g.nodes[neighbor]['keys'] is not None:
                        hd_sum += self.hamming_distance(subset, self.encode_keys(g.nodes[neighbor]['keys']))

                if hd_sum > max_hd_sum:
                    max_hd_sum = hd_sum
                    subsets_with_max_hd = [subset]
                elif hd_sum == max_hd_sum:
                    subsets_with_max_hd.append(subset)

            # print(f'subsets_with_max_hd (hd_sum={hd_sum}, max_hd_sum={max_hd_sum}, total={len(subsets_with_max_hd)}): {subsets_with_max_hd}')
            return subsets_with_max_hd

        def pick_best_subset_new_first(g: nx.Graph, node: int, possible_subsets_encoded: List[str]) -> str:
            """
            1. 对于节点 keys 为 None 的情况，视与其汉明距离为 0.
            2. 调用 find_subsets_with_max_hd_sum() 获得备选 subsets。
            3. 在备选 subsets 中，选出还未被使用过的 subsets，作为新的备选 subsets。如果全都使用过，则随机选择一个 subset 返回。
            4. 在新的备选 subsets 中，随机选择一个 subset 返回。

            1. for the case where node keys is None, treat the distance to the Hamming as 0.
            2. call find_subsets_with_max_hd_sum() to get possible subsets.
            3. among the possible subsets, select the unused subsets as the new possible subsets, if all of them have been used, select a subset randomly and return it.
            4. choose a random subset among the new possible subsets and return it.
            """

            # TODO: When subset_size = 1/2 * key_pool_size, there is a situation where neighbouring nodes are identical. The reason is to maximise max_hd_sum.

            subsets_with_max_hd = find_subsets_with_max_hd_sum(g, node, possible_subsets_encoded)
            unused_subsets = [subset for subset in subsets_with_max_hd if subset not in self.used_key_subsets]
            # print(f'unused_subsets (total={len(unused_subsets)}): {unused_subsets}')
            best_subset = random.choice(unused_subsets) if unused_subsets else random.choice(subsets_with_max_hd)
            # print(f'best_subset: {self.decode_keys(best_subset)}')
            self.used_key_subsets.add(best_subset)

            return best_subset

        def pick_best_subset_with_avg_keyusage(g: nx.Graph, node: int, possible_subsets_encoded: List[str]) -> str:
            """
            1. for the case where node keys is None, treat the distance to the Hamming as 0.
            2. call find_subsets_with_max_hd_sum() to get possible subsets.
            3. among the alternative subsets, choose the subset with the most even key_usage (so that each key is used as many times as possible)

            The method was not effective. It was not used in the simulation!
            """

            subsets_with_max_hd = find_subsets_with_max_hd_sum(g, node, possible_subsets_encoded)
            best_subset = min(subsets_with_max_hd, key=lambda x: max(self.key_usage[key] for key in self.decode_keys(x)))

            # print(f'best_subset: {self.decode_keys(best_subset)}')

            for key in self.decode_keys(best_subset):
                self.key_usage[key] += 1

            # print(f'key_usage: {self.key_usage}\n')

            return best_subset

        assigned_group = set()
        while len(assigned_group) < len(subgraph.nodes):
            if not assigned_group:
                node = max(non_source_nodes, key=lambda x: subgraph.degree[x])
            else:
                node = find_neighbor_of_group_with_max_degree(subgraph, assigned_group)

            # print(f'Selected node: {node} (degree: {subgraph.degree[node]}, assigned_group: {assigned_group})')

            if strategy == 'nf':
                best_subset_encoded = pick_best_subset_new_first(subgraph, node, possible_subsets_encoded)
            elif strategy == 'avg':
                best_subset_encoded = pick_best_subset_with_avg_keyusage(subgraph, node, possible_subsets_encoded)

            # print('\n')
            subgraph.nodes[node]['keys'] = self.decode_keys(best_subset_encoded)
            assigned_group.add(node)


class Attacker:
    def __init__(self, graph: NGraph, kdc: KDC):
        self.graph = graph
        self.kdc = kdc
        self.key_pool_compromised: Set[str] = set()
        self.finite_field_size = 2 ** 8

    def compromise_node_randomly_on_path(self, num_compromised_nodes: int) -> List[int]:
        """
        Compromises a fixed number of nodes randomly along the path.
        """

        if num_compromised_nodes > len(self.graph.picked_path) - 2:
            raise ValueError("num_compromised_nodes cannot be larger than number of nodes in the path")

        # only compromise intermediate nodes
        compromised_nodes = random.sample(self.graph.picked_path[1:-1], num_compromised_nodes)
        # print(f'compromised nodes: {compromised_nodes}')

        for node in compromised_nodes:
            self.graph.nxg.nodes[node]['compromised'] = True
            self.key_pool_compromised.update(self.graph.nxg.nodes[node]['keys'])

        return compromised_nodes

    def compromise_node(self, node: int):
        """
        Compromises a specific node.
        """

        if node not in self.graph.nxg.nodes:
            raise ValueError("Node not found in the graph")
        elif self.graph.nxg.nodes[node]['node_type'] != 'intermediate':
            raise ValueError("Node is not an intermediate node")

        self.graph.nxg.nodes[node]['compromised'] = True
        self.key_pool_compromised.update(self.graph.nxg.nodes[node]['keys'])

    def reset_comromised_status(self):
        for node in self.graph.nxg.nodes():
            self.graph.nxg.nodes[node]['compromised'] = False
        self.key_pool_compromised.clear()

    def run_single_attack(self, attack_type: str, distr_strategy: str, attack_model: str, subset_size: int, num_compromised_nodes: int):
        self.kdc.reset_keys()

        if distr_strategy == 'rd':
            self.kdc.distribute_fix_num_keys_randomly(subset_size)
        elif distr_strategy == 'mhd_n':
            self.kdc.distribute_fix_num_keys_with_max_hamming_distance(subset_size, strategy='nf')
        elif distr_strategy == 'mhd_a':
            self.kdc.distribute_fix_num_keys_with_max_hamming_distance(subset_size, strategy='avg')

        self.reset_comromised_status()
        compromised_nodes = self.compromise_node_randomly_on_path(num_compromised_nodes)

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
                        d = len(self.graph.nxg.nodes[nd]['keys'] - self.key_pool_compromised)
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

                is_success = 1
                latest_compromised_node = next((node for node in self.graph.picked_path if self.graph.nxg.nodes[node]['compromised']), None)

                for node in self.graph.picked_path[self.graph.picked_path.index(latest_compromised_node):]:
                    if self.graph.nxg.nodes[node]['compromised']:
                        latest_compromised_node = node
                        # print(f'Reached compromised node {node}')
                    else:
                        d = len(self.graph.nxg.nodes[node]['keys'] - self.graph.nxg.nodes[latest_compromised_node]['keys'])
                        if random.randint(1, self.finite_field_size ** d) != 1:
                            # if random.randint(1, 4) == 1:
                            is_success = 0
                            # print(f'Failed at node {node}')
                            break

                        # print(f'Success at node {node}')

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
                        compromised_key = random.choice(list(self.kdc.key_pool))
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
                    compromised_key = random.choice(list(self.kdc.key_pool))

                    for nd in path_from_c[1:]:  # Skip compromised node
                        if not self.graph.nxg.nodes[nd]['compromised']:
                            if compromised_key in self.graph.nxg.nodes[nd]['keys']:
                                path_success = 0
                                break

                    if path_success == 1:
                        is_success = 1
                        break

        return compromised_nodes, is_success


def demo():
    g1 = NGraph()
    g1.generate_ws_graph(num_nodes=12)

    # kdc = KDC(graph=g1, key_pool_size=8)
    # attacker = Attacker(graph=g1, kdc=kdc)
    #
    # kdc.reset_keys()    # don't forget to reset keys before distributing new keys
    # kdc.distribute_fix_num_keys_with_max_hamming_distance(key_subset_size=2, strategy='nf')
    # kdc.distribute_fix_num_keys_with_max_hamming_distance(key_subset_size=4, strategy='avg')
    # kdc.distribute_fix_num_keys_randomly(key_subset_size=3)

    g1.pick_path()
    # attacker.compromise_node_randomly_on_path(num_compromised_nodes=2)
    g1.take_snapshot(filename='/Users/xingyuzhou/TUNC/networkAnalysis/networks/temp.csv')
    g1.visualize()


def run_simulation(batch, batch_runs):
    g1 = NGraph()
    g1.restore_snapshot(filename='/Users/xingyuzhou/TUNC/networkAnalysis/networks/12n.csv')

    m_result_folder = None
    m_result_folder = '/Users/xingyuzhou/TUNC/networkAnalysis/tpa10w'
    m_key_pool_size = 12
    m_max_num_compromised_nodes = 3

    kdc = KDC(graph=g1, key_pool_size=m_key_pool_size)
    attacker = Attacker(graph=g1, kdc=kdc)

    g1.pick_path()

    m_attack_type = 'tpa'       # tpa | dpa
    m_distr_strategy = 'mhd_n'  # rd | mhd_n | mhd_a
    m_attack_model = 'other'      # our | other

    compromised_node_options = [i for i in range(1, m_max_num_compromised_nodes + 1)]
    subset_size_options = [i for i in range(1, kdc.key_pool_size)]

    compromised_node_options = [3]
    subset_size_options = [5, 6]

    if m_result_folder is not None:
        os.makedirs(m_result_folder, exist_ok=True)

    m_runs = batch_runs * 10000
    for num_compromised_nodes in compromised_node_options:
        for subset_size in subset_size_options:
            results = []
            for _ in tqdm(range(m_runs), desc=f'Running {m_attack_type} ({m_distr_strategy}, {m_attack_model}) for c=({num_compromised_nodes}/{m_max_num_compromised_nodes}), keys=({subset_size}/{len(subset_size_options)})'):
                compromised_nodes, is_success = attacker.run_single_attack(m_attack_type, m_distr_strategy, m_attack_model, subset_size, num_compromised_nodes)
                results.append([compromised_nodes, is_success])

            if m_result_folder is not None:
                result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success'])
                result.to_csv(f'{m_result_folder}/csv{batch}_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c_{subset_size}keys.csv', index=False)


# def run_simulation_parallel():
#     g1 = NGraph()
#     g1.restore_snapshot(filename='/Users/xingyuzhou/TUNC/networkAnalysis/networks/12n.csv')
#
#     m_result_folder = None
#     m_result_folder = '/Users/xingyuzhou/TUNC/networkAnalysis/ppp'
#     m_key_pool_size = 12
#     m_runs = 10
#     m_max_num_compromised_nodes = 3
#
#     kdc = KDC(graph=g1, key_pool_size=m_key_pool_size)
#     attacker = Attacker(graph=g1, kdc=kdc)
#
#     g1.pick_path()
#
#     m_attack_type = 'tpa'       # tpa | dpa
#     m_distr_strategy = 'mhd_n'  # rd | mhd_n | mhd_a
#     m_attack_model = 'our'      # our | other
#
#     cores = 4  # Number of cores on Apple M2
#     partitioned_runs = m_runs // cores
#
#     if m_result_folder is not None:
#         os.makedirs(m_result_folder, exist_ok=True)
#
#     for num_compromised_nodes in range(1, m_max_num_compromised_nodes + 1):
#         for subset_size in range(1, kdc.key_pool_size):
#             with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
#                 futures = [executor.submit(wraper_run_single_attack, attacker, m_attack_type, m_distr_strategy, m_attack_model, subset_size, num_compromised_nodes) for _ in range(partitioned_runs)]
#                 results = []
#
#                 for future in concurrent.futures.as_completed(futures):
#                     results.append(future.result())
#
#             if m_result_folder is not None:
#                 result = pd.DataFrame(results, columns=['compromised_nodes', 'is_success'])
#                 result.to_csv(f'{m_result_folder}/csv_{m_attack_type}_{m_distr_strategy}_{m_attack_model}_{m_runs}runs_{num_compromised_nodes}c_{subset_size}keys.csv', index=False)
#
#
# def wraper_run_single_attack(attacker, m_attack_type, m_distr_strategy, m_attack_model, subset_size, num_compromised_nodes):
#     compromised_nodes, is_success = attacker.run_single_attack(m_attack_type, m_distr_strategy, m_attack_model, subset_size, num_compromised_nodes)
#     return [compromised_nodes, is_success]


def merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, subset_size):
    result_folder = '/Users/xingyuzhou/TUNC/networkAnalysis/tpa10w'

    # List to store dataframes from all batches.
    dfs = []
    count_runs = 0

    for file in os.listdir(result_folder):
        # Check for required elements in the file name.
        if re.match(r"csv\d+_", file) and file.endswith(".csv") and all(x in file for x in [attacker_type, distr_strategy, attack_model, f'{num_compromised_nodes}c', f'{subset_size}keys']):

            # Extract the number of runs from the file name.
            match = re.search(r'(\d+)runs', file)
            if match:
                runs_in_file = int(match.group(1))
                count_runs += runs_in_file

            df = pd.read_csv(os.path.join(result_folder, file))
            dfs.append(df)

    if not dfs:
        return

    # Combine all dataframes.
    combined_df = pd.concat(dfs, ignore_index=True)

    # Generate the combined CSV file name.
    combined_csv_name = f'csv_{attacker_type}_{distr_strategy}_{attack_model}_{count_runs}runs_{num_compromised_nodes}c_{subset_size}keys.csv'
    combined_df.to_csv(os.path.join(result_folder, combined_csv_name), index=False)
    print(f'Combined {len(dfs)} CSV files into {combined_csv_name}')

if __name__ == "__main__":
    # demo()

    for attacker_type in ['tpa', 'dpa']:
        for distr_strategy in ['rd', 'mhd_n', 'mhd_a']:
            for attack_model in ['our', 'other']:
                for num_compromised_nodes in range(1, 4):
                    for subset_size in range(1, 12):
                        merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, subset_size)

    # parser = argparse.ArgumentParser(description='Run simulation with specified batch number')
    # parser.add_argument('batch', type=int, help='Batch number for the simulation')
    # parser.add_argument('batch_runs', type=int, help='Number of runs in the batch (* 100000)')
    # args = parser.parse_args()
    # run_simulation(batch=args.batch, batch_runs=args.batch_runs)      # for example, python3 pr_new.py 1 1
