import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd

from typing import Set, List, Dict, Tuple
from itertools import combinations
from tqdm import tqdm


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

        df_combined = pd.concat([df_rm, df_attrs], axis=1) # axis=1: concat by column
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
            self.nxg.nodes[node]['v_color']: str = None

    def visualize(self):
        num_colors = 100
        colors = plt.colormaps['tab20']
        color_map = {i: colors(i) for i in range(num_colors)}

        node_sizes = {}
        node_labels = {}
        v_colors = [color_map.get(self.nxg.nodes[node]['v_color'], 'grey') for node in self.nxg.nodes]

        for node in self.nxg.nodes:
            node_type = self.nxg.nodes[node]['node_type']

            if node_type in ['source', 'destination']:
                node_sizes[node] = 800
            elif self.nxg.nodes[node]['compromised']:
                node_sizes[node] = 600
            else:
                node_sizes[node] = 300  # Default size for other nodes

            node_labels[node] = f"{node_type[0].upper()}{node}\n{self.nxg.nodes[node]['keys']}"

        plt.figure(figsize=(12, 12))
        pos = nx.kamada_kawai_layout(self.nxg)

        compromised_nodes = [node for node in self.nxg.nodes if self.nxg.nodes[node]['compromised']]
        normal_nodes = [node for node in self.nxg.nodes if not self.nxg.nodes[node]['compromised']]

        nx.draw_networkx_nodes(self.nxg, pos, nodelist=normal_nodes, node_size=[node_sizes[n] for n in normal_nodes], node_color=[v_colors[n] for n in normal_nodes])
        nx.draw_networkx_nodes(self.nxg, pos, nodelist=compromised_nodes, node_size=[node_sizes[n] for n in compromised_nodes], node_color=[v_colors[n] for n in compromised_nodes], linewidths=1.5, edgecolors='red', node_shape='*')

        nx.draw_networkx_edges(self.nxg, pos, edgelist=self.nxg.edges, edge_color='black', width=1, alpha=0.3)

        if self.picked_path is not None:
            nx.draw_networkx_edges(self.nxg, pos, edgelist=[(self.picked_path[i], self.picked_path[i + 1]) for i in range(len(self.picked_path) - 1)], edge_color='red', width=1.5)

        nx.draw_networkx_labels(self.nxg, pos, labels=node_labels)

        plt.show()
        plt.close()

    def pick_path(self):
        source = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 'source']
        destination = [node for node, attr in self.nxg.nodes(data=True) if attr.get('node_type') == 'destination']

        if not source or not destination:
            print("Source or destination node not found.")
            return None

        source, destination = source[0], destination[0]

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

    def distribute_fix_num_keys_randomly(self, key_subset_size: int):
        """
        Distributes a fixed number of keys randomly to each non-source node.
        """

        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 'source']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        subsets_encoded_to_int = {}     # for coloring

        for node in subgraph.nodes:
            subgraph.nodes[node]['keys'] = set(random.sample(sorted(self.key_pool), key_subset_size))
            if self.encode_keys(subgraph.nodes[node]['keys']) not in subsets_encoded_to_int:
                subsets_encoded_to_int[self.encode_keys(subgraph.nodes[node]['keys'])] = len(subsets_encoded_to_int)

            subgraph.nodes[node]['v_color'] = subsets_encoded_to_int[self.encode_keys(subgraph.nodes[node]['keys'])]

    def reset_keys(self):
        for node in self.graph.nxg.nodes():
            self.graph.nxg.nodes[node]['keys'] = None

    def color_greedy(self):
        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 'source']
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        coloring = nx.greedy_color(subgraph, strategy='largest_first')
        total_colors = len(set(coloring.values()))

        for node, color in coloring.items():
            self.graph.nxg.nodes[node]['v_color'] = color

        return total_colors

    def encode_keys(self, node_keys):
        return ''.join(['1' if f'k{i}' in node_keys else '0' for i in range(self.key_pool_size)])

    def decode_keys(self, encoded_keys):
        return {f'k{i}' for i in range(self.key_pool_size) if encoded_keys[i] == '1'}

    def hamming_distance(self, str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def distribute_fix_num_keys_with_max_hamming_distance(self, key_subset_size: int):
        # Encode each node's key set into binary strings

        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        possible_subsets = list(combinations(self.key_pool, key_subset_size))
        # print(f'possible subsets: {possible_subsets}\n')
        possible_subsets_encoded = [self.encode_keys(key_subset) for key_subset in possible_subsets]
        # print(f'possible subsets encoded: {possible_subsets_encoded}\n')

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 'source']
        # subgraph can change attributes of the original graph, but can not change layout of the original graph
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        # nx.draw(subgraph, with_labels=True)

        sort_result = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)
        nodes_degree_desc = [node for node, degree in sort_result]

        subsets_encoded_to_int = {}     # for coloring

        for node in nodes_degree_desc:
            max_hamming_distance_sum = 0
            best_subset_encoded = None

            for subset_encoded in possible_subsets_encoded:
                hamming_distance_sum = 0
                valid_neighbor = 0

                for neighbor in subgraph.neighbors(node):
                    if subgraph.nodes[neighbor]['keys'] is not None:
                        neighbor_subset_encoded = self.encode_keys(subgraph.nodes[neighbor]['keys'])
                        hamming_distance_sum += self.hamming_distance(subset_encoded, neighbor_subset_encoded)
                        valid_neighbor += 1
                    else:
                        neighbor_subset_encoded = None

                    # print(f'node: {node}, current: {subset_encoded}, neighbor: {neighbor} ({neighbor_subset_encoded}), hd sum: {hamming_distance_sum}')

                if self.key_pool_size >= 2 * key_subset_size:
                    temp_theo_hmd_sum = key_subset_size * 2 * valid_neighbor
                else:
                    temp_theo_hmd_sum = 2 * (self.key_pool_size - key_subset_size) * valid_neighbor

                if hamming_distance_sum == temp_theo_hmd_sum:
                    best_subset_encoded = subset_encoded
                    # print(f'reached temp theoretical max hdm: {temp_theo_hmd_sum}')
                    break
                elif hamming_distance_sum > max_hamming_distance_sum:
                    max_hamming_distance_sum = hamming_distance_sum
                    best_subset_encoded = subset_encoded
                    # print(f'cannt reach temp theoretical max hdm ({hamming_distance_sum} < {temp_theo_hmd_sum})')

            # print(f'best subset: {best_subset_encoded}\n')

            if best_subset_encoded not in subsets_encoded_to_int:
                subsets_encoded_to_int[best_subset_encoded] = len(subsets_encoded_to_int)

            subgraph.nodes[node]['keys'] = self.decode_keys(best_subset_encoded)
            subgraph.nodes[node]['v_color'] = subsets_encoded_to_int[best_subset_encoded]

        # print(f'color used: {len(subsets_encoded_to_int)}')

        improvable = True
        for node in subgraph.nodes:

            if self.key_pool_size >= 2 * key_subset_size:
                theo_max_hmd_sum = key_subset_size * 2 * subgraph.degree[node]
            else:
                theo_max_hmd_sum = 2 * (self.key_pool_size - key_subset_size) * subgraph.degree[node]

            hd_sum = 0
            for neighbor in subgraph.neighbors(node):
                hd_sum += self.hamming_distance(self.encode_keys(subgraph.nodes[node]['keys']), self.encode_keys(subgraph.nodes[neighbor]['keys']))

            if hd_sum != theo_max_hmd_sum:
                improvable = False
                # print(f"Node {node} did not achieve the theoretical max hamming distance sum ({hd_sum} < {theo_max_hmd_sum})")

        if improvable:
            # print("Consider to reduce key_pool_size")
            pass

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

    def dpa_random_pos_on_path(self, runs: int, max_num_compromised_nodes: int, distr_strategy: str):
        df_columns = ['compromised_nodes', 'is_success']

        subset_sizes = [i for i in range(1, self.kdc.key_pool_size)]
        # subset_sizes = [1, 2, 3, 4, 5]

        for num_compromised_nodes in range(1, max_num_compromised_nodes + 1):
            for subset_size in subset_sizes:
                result = pd.DataFrame(columns=df_columns)

                for _ in tqdm(range(runs), desc=f'Running for c=({num_compromised_nodes}/{max_num_compromised_nodes}), keys=({subset_size}/{len(subset_sizes)})'):
                    self.kdc.reset_keys()
                    if distr_strategy == 'rd':
                        self.kdc.distribute_fix_num_keys_randomly(subset_size)
                    elif distr_strategy == 'mhd':
                        self.kdc.distribute_fix_num_keys_with_max_hamming_distance(subset_size)

                    self.reset_comromised_status()
                    compromised_nodes = self.compromise_node_randomly_on_path(num_compromised_nodes)

                    path_from_c_to_d = []
                    for c_node in compromised_nodes:
                        path_from_c_to_d.append(self.graph.picked_path[self.graph.picked_path.index(c_node):])

                    for pt in path_from_c_to_d:
                        is_success = 0  # 0: fail, 1: success
                        for nd in pt[1:]:   # skip the first node, which is the compromised node
                            # print(f'compromised nodes: {compromised_nodes} ({self.key_pool_compromised}), current node: {nd} ({self.graph.nxg.nodes[nd]["keys"]})')
                            if self.graph.nxg.nodes[nd]['keys'] <= self.key_pool_compromised:   # all keys are compromised
                                is_success = 1
                                # print('success (all keys compromised)')
                            else:
                                d = len(self.graph.nxg.nodes[nd]['keys'] - self.key_pool_compromised)  # number of uncompromised keys
                                if random.randint(1, self.finite_field_size ** d) == 1:
                                    is_success = 1
                                    # print('success (randomly guessed)')
                                else:
                                    is_success = 0
                                    # print('fail')
                                    break

                        if is_success == 1:     # if success, no need to check other paths
                            break

                    new_row = pd.DataFrame([[compromised_nodes, is_success]], columns=df_columns)
                    result = pd.concat([result, new_row], ignore_index=True)

                result.to_csv(f'networkAnalysis/results/dpa_random_pos/csv_{distr_strategy}_{runs}runs_{num_compromised_nodes}c_{subset_size}keys.csv', index=False)


def demo():
    g1 = NGraph()
    g1.restore_snapshot(f'graph_ws_12nodes.csv')

    kdc = KDC(graph=g1, key_pool_size=8)
    attacker = Attacker(graph=g1, kdc=kdc)

    kdc.reset_keys()    # don't forget to reset keys before distributing new keys
    # kdc.distribute_fix_num_keys_with_max_hamming_distance(key_subset_size=3)
    kdc.distribute_fix_num_keys_randomly(key_subset_size=3)

    g1.pick_path()
    attacker.compromise_node_randomly_on_path(num_compromised_nodes=2)
    g1.take_snapshot(filename='graph_temp2.csv')
    g1.visualize()


def run_simulation():
    g1 = NGraph()
    g1.restore_snapshot(f'graph_ws_12nodes.csv')

    kdc = KDC(graph=g1, key_pool_size=8)
    attacker = Attacker(graph=g1, kdc=kdc)

    g1.pick_path()

    for distr_strategy in ['rd', 'mhd']:
        attacker.dpa_random_pos_on_path(runs=1000, max_num_compromised_nodes=3, distr_strategy=distr_strategy)


if __name__ == "__main__":
    demo()
