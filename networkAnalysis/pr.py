import random
from typing import Set, List

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm


class Node:
    def __init__(self, node_id, node_type):
        self.node_id: str = node_id
        self.node_type: str = node_type
        self.keys: Set[str] = set()
        self.neighbours: Set[Node] = set()
        self.compromised: bool = False
        self.color: int = -1

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

    def generate_random_network(self, pr: float):
        if pr < 0 or pr > 1:
            raise ValueError("Probability p must be between 0 and 1")

        while True:
            # generate a random graph using Erdős–Rényi model
            g = nx.random_graphs.erdos_renyi_graph(len(self.nodes), pr)

            # map NetworkX node IDs to our node IDs
            id_mapping = {i: node.node_id for i, node in enumerate(self.nodes)}
            g = nx.relabel_nodes(g, id_mapping)

            # check if the generated graph is valid
            if self.is_graph_valid(g):
                break

        # add nx graph edges to our graph
        for edge in g.edges():
            self.add_edge(edge[0], edge[1])

    def is_graph_valid(self, g):
        sources = [node for node in g.nodes if node.startswith('S')]
        sinks = [node for node in g.nodes if node.startswith('D')]
        intermediates = [node for node in g.nodes if node.startswith('I')]

        node_in_path = set()

        for source_node in sources:
            for sink_node in sinks:
                paths = list(nx.all_simple_paths(g, source=source_node, target=sink_node))

                for path in paths:
                    # each path must have at least 1 intermediate node
                    if len(path) < 3:
                        return False

                    # source and sink nodes can only be at the beginning and end of the path
                    intermediate_nodes = path[1:-1]
                    if any(node not in intermediates for node in intermediate_nodes):
                        return False

                    node_in_path.update(path)

        # no node is isolated
        return node_in_path == set(g.nodes)

    def create_reachability_matrix(self, filename='reachability_matrix.csv'):
        # sort nodes by type and ID
        nodes_sorted = sorted(self.nodes, key=lambda n: (n.node_type, n.node_id))

        # initialize matrix
        matrix = pd.DataFrame(0, index=[n.node_id for n in nodes_sorted], columns=[n.node_id for n in nodes_sorted])

        # fill matrix
        for node in nodes_sorted:
            for neighbour in node.neighbours:
                matrix.at[node.node_id, neighbour.node_id] = 1

        # save matrix to CSV file
        matrix.to_csv(filename)

        return matrix

    def color_greedy(self):
        for node in self.nodes:
            if node.node_type == "source":  # skip source nodes
                continue

            available_colors = set(range(len(self.nodes)))
            for neighbor in node.neighbours:
                if neighbor.color != -1:
                    available_colors.discard(neighbor.color)

            # 分配最小可用颜色
            node.color = min(available_colors)

        # 计算使用的颜色数
        num_colors = len(set(node.color for node in self.nodes if node.node_type != "source"))
        return num_colors

    def color_welsh_powell(self):
        # 按度降序排列非源节点
        sorted_nodes = sorted([node for node in self.nodes if node.node_type != "source"], key=lambda node: len(node.neighbours), reverse=True)

        # 为每个节点分配颜色
        for node in sorted_nodes:
            available_colors = [True] * len(self.nodes)
            for neighbor in node.neighbours:
                if neighbor.color != -1:
                    available_colors[neighbor.color] = False

            # 寻找第一个可用的颜色
            color = available_colors.index(True)
            node.color = color

        # 计算使用的颜色数
        num_colors = len(set(node.color for node in sorted_nodes))
        return num_colors

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

        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.show()

        plt.close()

    def visualize_colored_graph(self):
        # 创建两个 NetworkX 图表示
        g_greedy = nx.Graph()
        g_welsh_powell = nx.Graph()

        # 应用贪心算法着色，并添加节点和边到 g_greedy
        num_colors_greedy = self.color_greedy()
        for node in self.nodes:
            g_greedy.add_node(node.node_id, color=node.color)
            for neighbour in node.neighbours:
                g_greedy.add_edge(node.node_id, neighbour.node_id)

        # 重置颜色，应用 Welsh-Powell 算法着色，并添加节点和边到 g_welsh_powell
        for node in self.nodes:
            node.color = -1

        num_colors_welsh_powell = self.color_welsh_powell()
        for node in self.nodes:
            g_welsh_powell.add_node(node.node_id, color=node.color)
            for neighbour in node.neighbours:
                g_welsh_powell.add_edge(node.node_id, neighbour.node_id)

        # 准备布局
        pos_greedy = nx.spring_layout(g_greedy)
        pos_welsh_powell = nx.spring_layout(g_welsh_powell)

        # 获取节点颜色
        colors_greedy = [g_greedy.nodes[node]['color'] for node in g_greedy.nodes]
        colors_welsh_powell = [g_welsh_powell.nodes[node]['color'] for node in g_welsh_powell.nodes]

        # 创建一个绘图区域，包含两个子图
        plt.figure(figsize=(12, 6))

        # 绘制使用贪心算法的结果
        plt.subplot(121)
        nx.draw(g_greedy, pos_greedy, with_labels=True, node_color=colors_greedy, cmap=plt.cm.viridis, node_size=500)
        plt.title(f'Greedy Algorithm Coloring\nNumber of colors used: {num_colors_greedy}')

        # 绘制使用 Welsh-Powell 算法的结果
        plt.subplot(122)
        nx.draw(g_welsh_powell, pos_welsh_powell, with_labels=True, node_color=colors_welsh_powell, cmap=plt.cm.plasma, node_size=500)
        plt.title(f'Welsh-Powell Algorithm Coloring\nNumber of colors used: {num_colors_welsh_powell}')

        plt.show()


class KDC:
    def __init__(self, graph, key_pool_size: int):
        self.graph = graph
        self.key_pool_size: int = key_pool_size
        self.key_pool: Set[str] = {f"k{i}" for i in range(key_pool_size)}
        self.max_key_subset_size: int = key_pool_size - 1

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
        return f"KDC({self.key_pool_size}, {self.key_pool})"


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
            self.key_pool_compromised.update(node.keys)     # Add compromised node's keys if not already in key_pool_compromised

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
            for key_subset_size in range(1, self.kdc.max_key_subset_size + 1):
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


def load_graph_from_matrix(filename='reachability_matrix.csv'):
    # 读取 CSV 文件
    matrix = pd.read_csv(filename, index_col=0)

    # 统计不同类型节点的数量
    node_counts = {'S': 0, 'I': 0, 'D': 0}
    for node_id in matrix.columns:
        node_type = node_id[0]  # 假设节点ID的第一个字符表示节点类型
        if node_type in node_counts:
            node_counts[node_type] += 1

    # 创建图对象
    graph = Graph(node_counts['S'], node_counts['I'], node_counts['D'])

    # 添加节点和边
    for i, row in matrix.iterrows():
        for j, cell in row.items():
            if cell == 1:
                # 检查节点是否已经存在，如果不存在，则添加
                if not graph.find_node_by_id(i):
                    graph.add_node(i[0])  # 节点类型是ID的第一个字符
                if not graph.find_node_by_id(j):
                    graph.add_node(j[0])

                graph.add_edge(i, j)

    return graph

def compare_graphs(graph1: Graph, graph2: Graph) -> bool:
    # 比较节点数量
    if len(graph1.nodes) != len(graph2.nodes):
        return False

    # 比较每个节点的类型和ID
    nodes1 = {node.node_id: node.node_type for node in graph1.nodes}
    nodes2 = {node.node_id: node.node_type for node in graph2.nodes}
    if nodes1 != nodes2:
        return False

    # 比较边
    for node in graph1.nodes:
        neighbours1 = {neighbour.node_id for neighbour in node.neighbours}
        corresponding_node_in_graph2 = graph2.find_node_by_id(node.node_id)
        if corresponding_node_in_graph2 is None:
            return False
        neighbours2 = {neighbour.node_id for neighbour in corresponding_node_in_graph2.neighbours}

        if neighbours1 != neighbours2:
            return False

    return True


if __name__ == "__main__":
    # specify following parameters
    sources = 1
    intermediates = 10
    sinks = 1
    pr = 0.1

    runs = 100000
    key_pool_size = 10
    max_num_compromised_nodes = 5

    graph1 = Graph(sources, intermediates, sinks)
    graph1.generate_random_network(pr)
    graph1.create_reachability_matrix(f'./networkAnalysis/RM_{sources}S_{intermediates}I_{sinks}D_{pr}pr.csv')

    # graph1.visualize(None)
    graph1.visualize_colored_graph()

    # graph2 = load_graph_from_matrix()
    # print(compare_graphs(graph1, graph2))

    # kdc = KDC(graph, key_pool_size)
    # attacker = Attacker(graph, kdc)

    # one_way = graph.generate_one_way_network()
    #
    # max_hop = len(one_way) - 2
    # max_key_subset_size = kdc.max_key_subset_size


    # attacker.dpa_random_pos(runs, max_num_compromised_nodes)
    # attacker.dpa_fix_pos(one_way, runs)

    # print(f"runs: {runs},\nmax_num_compromised_nodes: {max_num_compromised_nodes},\nhops: {max_hop},\nmax_key_subset_size: {max_key_subset_size}")
