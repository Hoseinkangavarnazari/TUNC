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
        self.node_attrs_count: int = 6
        self.nxg = nx.Graph()
        self.picked_path: List[int] = None

    def generate_ws_graph(self, num_nodes: int):
        self.type = 'ws'

        # k represents each node's degree, it should be even and greater than 2
        # p represents the probability of rewiring. Small p will result in a graph that is more similar to a ring lattice, while large p will result in a graph that is more similar to a random graph.
        self.nxg = nx.watts_strogatz_graph(num_nodes, k=4, p=0.5)

        while not nx.is_connected(self.nxg):
            self.nxg = nx.watts_strogatz_graph(num_nodes, k=4, p=0.5)

        self.add_default_attrs()

        nodes_list = list(self.nxg.nodes)

        source, sink = random.sample(nodes_list, 2)
        while self.nxg.has_edge(source, sink):
            source, sink = random.sample(nodes_list, 2)

        self.nxg.nodes[source]['node_type'] = 'source'
        self.nxg.nodes[sink]['node_type'] = 'destination'

        for node in self.nxg.nodes:
            if self.nxg.nodes[node]['node_type'] is None:
                self.nxg.nodes[node]['node_type'] = 'intermediate'

    def save_graph(self):
        """
        Take a snapshot of the current graph, including reachability matrix and node attributes.
        """

        # filename = f'graph_{self.type}_{self.nxg.number_of_nodes()}nodes.csv'
        filename = f'graph_temp.csv'

        df_rm = nx.to_pandas_adjacency(self.nxg, dtype=int)

        attrs = {node: self.nxg.nodes[node] for node in self.nxg.nodes}
        df_attrs = pd.DataFrame.from_dict(attrs, orient='index')

        df_combined = pd.concat([df_rm, df_attrs], axis=1) # axis=1: concat by column
        df_combined.to_csv(filename)

    def load_graph(self, filename):
        df_combined = pd.read_csv(filename, index_col=0)    # index_col=0: use the first column as index

        df_rm = df_combined.iloc[:, :df_combined.shape[1] - self.node_attrs_count]  # extract reachability matrix
        df_rm.columns = df_rm.columns.map(int)  # convert column names to int
        self.nxg = nx.from_pandas_adjacency(df_rm)

        df_attrs = df_combined.iloc[:, -self.node_attrs_count:]     # extract node attributes
        for node in self.nxg.nodes:
            for attr in df_attrs.columns:
                self.nxg.nodes[node][attr] = df_attrs.loc[node, attr]

    def add_default_attrs(self):
        # update self.node_attrs_count if new attributes are added
        for node in self.nxg.nodes:
            self.nxg.nodes[node]['node_type']: str = None
            self.nxg.nodes[node]['compromised']: bool = False
            self.nxg.nodes[node]['keys']: Set[str] = None
            self.nxg.nodes[node]['v_label']: str = None
            self.nxg.nodes[node]['v_size']: int = 100
            self.nxg.nodes[node]['v_color']: str = None

    def visualize(self):
        num_colors = 100
        colors = plt.colormaps['tab20']
        color_map = {i : colors(i) for i in range(num_colors)}

        for node in self.nxg.nodes:
            node_type = self.nxg.nodes[node]['node_type']
            compromised_flag = ' (C)' if self.nxg.nodes[node]['compromised'] else ''
            self.nxg.nodes[node]['v_label'] = node_type[0].upper() + str(node) + compromised_flag + '\n' + str(self.nxg.nodes[node]['keys'])

            if node_type == 'source' or node_type == 'destination':
                self.nxg.nodes[node]['v_size'] = 500

        plt.figure(figsize=(12, 12))
        pos = nx.kamada_kawai_layout(self.nxg)

        v_colors = [color_map.get(self.nxg.nodes[node]['v_color'], 'grey') for node in self.nxg.nodes]
        v_sizes = [self.nxg.nodes[node]['v_size'] for node in self.nxg.nodes]
        v_labels = {node: self.nxg.nodes[node]['v_label'] for node in self.nxg.nodes}

        nx.draw_networkx_nodes(self.nxg, pos, node_size=v_sizes, node_color=v_colors)
        nx.draw_networkx_edges(self.nxg, pos, edgelist=self.nxg.edges, edge_color='black', width=1)

        if self.picked_path is not None:
            nx.draw_networkx_edges(self.nxg, pos, edgelist=[(self.picked_path[i], self.picked_path[i + 1]) for i in range(len(self.picked_path) - 1)], edge_color='red', width=2)

        nx.draw_networkx_labels(self.nxg, pos, labels=v_labels)

        # plt.axis('off')
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
        Distributes a fixed number of keys randomly to each node.
        """

        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        for node in self.graph.nxg.nodes():
            # source nodes have all the keys
            if self.graph.nxg.nodes[node]['node_type'] == "source":
                self.graph.nxg.nodes[node]['keys'] = self.key_pool.copy()
            else:
                self.graph.nxg.nodes[node]['keys'] = set(random.sample(sorted(self.key_pool), key_subset_size))

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

    def distribute_fix_num_keys_with_max_hamming_distance(self, key_subset_size: int):
        # Encode each node's key set into binary strings
        def encode_keys(node_keys):
            return ''.join(['1' if f'k{i}' in node_keys else '0' for i in range(self.key_pool_size)])

        def decode_keys(encoded_keys):
            return {f'k{i}' for i in range(self.key_pool_size) if encoded_keys[i] == '1'}

        def hamming_distance(str1, str2):
            return sum(c1 != c2 for c1, c2 in zip(str1, str2))

        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        possible_subsets = list(combinations(self.key_pool, key_subset_size))
        possible_subsets_encoded = [encode_keys(key_subset) for key_subset in possible_subsets]
        # print(f'possible subsets: {possible_subsets_encoded}\n')

        non_source_nodes = [node for node in self.graph.nxg.nodes if self.graph.nxg.nodes[node]['node_type'] != 'source']
        # subgraph can change attributes of the original graph, but can not change layout of the original graph
        subgraph = self.graph.nxg.subgraph(non_source_nodes)

        # nx.draw(subgraph, with_labels=True)

        sort_result = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)
        nodes_degree_desc = [node for node, degree in sort_result]

        subsets_encoded_to_int = {}

        for node in nodes_degree_desc:
            max_hamming_distance_sum = 0
            best_subset_encoded = None

            for subset_encoded in possible_subsets_encoded:
                hamming_distance_sum = 0
                valid_neighbor = 0

                for neighbor in subgraph.neighbors(node):
                    if subgraph.nodes[neighbor]['keys'] is not None:
                        neighbor_subset_encoded = encode_keys(subgraph.nodes[neighbor]['keys'])
                        hamming_distance_sum += hamming_distance(subset_encoded, neighbor_subset_encoded)
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

            subgraph.nodes[node]['keys'] = decode_keys(best_subset_encoded)
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
                hd_sum += hamming_distance(encode_keys(subgraph.nodes[node]['keys']), encode_keys(subgraph.nodes[neighbor]['keys']))

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

        # for num_compromised_nodes in tqdm(range(1, max_num_compromised_nodes + 1), desc='Overall Progress'):
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



if __name__ == "__main__":
    total_nodes = 10
    key_pool_size = 8
    max_num_compromised_nodes = 3
    runs = 100000
    # distr_strategy = 'max_hamming'  # 'random' or 'max_hamming'
    distr_strategies = ['rd', 'mhd']

    g1 = NGraph()
    # g1.generate_ws_graph(total_nodes)
    g1.load_graph(f'graph_ws_12nodes.csv')
    kdc = KDC(g1, key_pool_size)
    attacker = Attacker(g1, kdc)

    g1.pick_path()
    for distr_strategy in distr_strategies:
        attacker.dpa_random_pos_on_path(runs, max_num_compromised_nodes, distr_strategy)

    # g1.visualize()
    # g1.save_graph()

    # attacker = Attacker(g, kdc)

    # g1.save_graph()
