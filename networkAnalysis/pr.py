import os
import random
import re

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm


class Node:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type
        self.keys = []
        self.downstreams = []
        self.compromised = False

    def add_edge(self, node):
        self.downstreams.append(node)

    def set_keys(self, keys):
        self.keys = keys

    def set_compromised(self, is_compromised):
        self.compromised = is_compromised

    def empty_keys(self):
        self.keys = []

    def __repr__(self):
        return f"Node({self.node_id}, {self.node_type}, {self.keys}, is_compromised={self.compromised})"


class Graph:
    def __init__(self, s_count, i_count, d_count):
        self.nodes = {}
        self.counters = {"source": 0, "intermediate": 0, "sink": 0}
        self.prefixes = {"source": "S", "intermediate": "I", "sink": "D"}

        for _ in range(s_count):
            self.add_node("source")
        for _ in range(i_count):
            self.add_node("intermediate")
        for _ in range(d_count):
            self.add_node("sink")

    def add_node(self, node_type):
        self.counters[node_type] += 1
        prefix = self.prefixes[node_type]
        node_id = f"{prefix}{self.counters[node_type]}"  # Generates S1, I1, D1, etc.
        new_node = Node(node_id, node_type)
        self.nodes[node_id] = new_node

    def add_edge(self, from_node_id, to_node_id):
        if from_node_id in self.nodes and to_node_id in self.nodes:
            self.nodes[from_node_id].add_edge(self.nodes[to_node_id])
        else:
            raise ValueError("One or both nodes not found in the graph")

    def generate_random_network(self):
        pass

    def generate_one_way_network(self):
        # Get all nodes by type
        source_nodes = [node for node in self.nodes.values() if node.node_type == "source"]
        intermediate_nodes = [node for node in self.nodes.values() if node.node_type == "intermediate"]
        sink_nodes = [node for node in self.nodes.values() if node.node_type == "sink"]

        # Ensure there is exactly one source and one sink
        if len(source_nodes) != 1 or len(sink_nodes) != 1:
            raise ValueError("Graph must have exactly one source and one sink node")

        # Start from the source node
        current_node = source_nodes[0]

        # Connect to each intermediate node in sequence
        for next_node in intermediate_nodes:
            self.add_edge(current_node.node_id, next_node.node_id)
            current_node = next_node

        # Connect the last intermediate node (or source if no intermediates) to the sink
        self.add_edge(current_node.node_id, sink_nodes[0].node_id)

    def find_path(self, from_node_id, to_node_id):
        def dfs(current_node, path):
            if current_node.node_id == to_node_id:
                return path
            for next_node in current_node.downstreams:
                result = dfs(next_node, path + [next_node.node_id])
                if result is not None:
                    return result
            return None

        start_node = self.nodes[from_node_id]
        return dfs(start_node, [from_node_id])

    def visualize(self, figname):
        plt.figure()

        colors = {
            'source': '#00FF004D',  # Green with 30% opacity
            'sink': '#FF00004D',  # Red with 30% opacity
            'compromised': '#FFFF004D',  # Yellow with 30% opacity
            'intermediate': '#0000FF4D'  # Blue with 30% opacity
        }

        G = nx.DiGraph()
        for node in self.nodes.values():
            G.add_node(node.node_id, label=f"{node.node_id}\n{' '.join(node.keys)}", color=colors['compromised'] if node.compromised else colors[node.node_type])
            for next_node in node.downstreams:
                G.add_edge(node.node_id, next_node.node_id)

        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos,
                with_labels=True,
                labels=nx.get_node_attributes(G, 'label'),
                node_color=nx.get_node_attributes(G, 'color').values())

        plt.savefig(figname, dpi=300)
        plt.close()


class KDC:
    def __init__(self, graph, key_pool_size):
        self.graph = graph
        self.key_pool_size = key_pool_size
        self.key_pool = [f"k{i}" for i in range(key_pool_size)]
        self.key_pool_size_options = list(range(1, key_pool_size))

    def distribute_keys(self, key_subset_size):
        if key_subset_size > self.key_pool_size:
            raise ValueError("key_subset_size cannot be larger than key_pool_size")

        for node in self.graph.nodes.values():
            if node.node_type == "source":
                node.set_keys(self.key_pool.copy())
            else:
                node.set_keys(random.sample(self.key_pool, key_subset_size))

    def reset_keys(self):
        for node in self.graph.nodes.values():
            node.empty_keys()

    def __repr__(self):
        return f"KDC({self.key_pool_size}, {self.key_pool}, {self.key_pool_size_options})"


class Attacker:
    def __init__(self, graph):
        self.graph = graph

    def compromise_node_random(self):
        intermediate_nodes = [node.node_id for node in self.graph.nodes.values() if node.node_type == "intermediate"]
        compromised_node_id = random.choice(intermediate_nodes)
        self.graph.nodes[compromised_node_id].set_compromised(True)
        return compromised_node_id

    def compromise_node_by_id(self, node_id):
        self.graph.nodes[node_id].set_compromised(True)

    def reset_compromised_nodes(self):
        for node in self.graph.nodes.values():
            node.set_compromised(False)


def data_pollution_attack(graph, kdc, runs):
    finite_field_size = 2 ** 8

    intermediate_nodes = [node.node_id for node in graph.nodes.values() if node.node_type == "intermediate"]
    sink_node_id = [node.node_id for node in graph.nodes.values() if node.node_type == "sink"][0]

    for key_subset_size in tqdm(kdc.key_pool_size_options, desc='Overall Progress'):
        df_columns = ["compromised_node_id", "path", "is_success"]
        results = pd.DataFrame(columns=df_columns)

        for _ in tqdm(range(runs), desc=f'Running attacks for key_subset_size={key_subset_size}'):
            kdc.reset_keys()
            kdc.distribute_keys(key_subset_size)

            compromised_node_id = random.choice(intermediate_nodes)
            graph.nodes[compromised_node_id].set_compromised(True)
            compromised_node = graph.nodes[compromised_node_id]
            path = graph.find_path(compromised_node_id, sink_node_id)

            if runs == 1:
                graph.visualize(f'networkAnalysis/network_{key_subset_size}.png')

            is_success = False
            for node_id in path[1:]:
                node = graph.nodes[node_id]
                if set(node.keys) == set(compromised_node.keys):
                    is_success = True
                else:
                    d = len(set(node.keys) - set(compromised_node.keys))    # Number of keys that node has but compromised_node doesn't
                    if random.randint(1, finite_field_size ** d) == 1:
                        is_success = True
                    else:
                        is_success = False
                        break

            new_row = pd.DataFrame([[compromised_node_id, path, is_success]], columns=df_columns)
            results = pd.concat([results, new_row], ignore_index=True)
            compromised_node.set_compromised(False)

        results.to_csv(f'networkAnalysis/dpa_results_{runs}_{key_subset_size}.csv', index=False)


def plot_results(folder_path, runs):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f'results_{runs}_' in f]
    csv_files.sort(key=lambda f: int(re.findall(r'\d+', f)[-1]))  # Sort by key_subset_size (ascending)
    key_subset_size = [int(re.findall(r'\d+', file)[-1]) for file in csv_files]
    print(f'{len(csv_files)} csv-files found; key_subset_size = {key_subset_size}')

    df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
    success_count = [(df['is_success'] == True).sum() for df in df_list]
    run_count = [len(df) for df in df_list]

    plt.figure()
    bars = plt.bar(key_subset_size, success_count, color='red', label='Success Count')
    plt.xlabel(f'key_subset_size (key pool size = {key_subset_size[-1] + 1})')
    plt.ylabel('Success Count')
    plt.title(f'Success Count (runs = {run_count[-1]})')
    plt.xticks(key_subset_size)
    plt.legend()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, yval, va='bottom', ha='center')

    plt.savefig(f'networkAnalysis/SuccessCount_{runs}.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    graph = Graph(1, 8, 1)
    kdc = KDC(graph, 10)

    # graph.generate_random_network()
    graph.generate_one_way_network()

    runs = 100000
    data_pollution_attack(graph, kdc, runs)
    plot_results('networkAnalysis', runs)
