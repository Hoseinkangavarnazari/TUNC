import numpy as np
from random import shuffle as sl
from random import randint as rd
import networkx as nx
import matplotlib.pyplot as plt


# node节点数量，edge边数量
def random_graph(node, edge):
    n = node
    node = range(0, n)
    node = list(node)

    sl(node)  # 生成拓扑排序
    m = edge
    result = []  # 存储生成的边，边用tuple的形式存储

    appeared_node = []
    not_appeared_node = node
    # 生成前n - 1条边
    while len(result) != n - 1:
        # 生成第一条边
        if len(result) == 0:
            p1 = rd(0, n - 2)
            p2 = rd(p1 + 1, n - 1)
            x = node[p1]
            y = node[p2]
            appeared_node.append(x)
            appeared_node.append(y)
            not_appeared_node = list(set(node).difference(set(appeared_node)))
            result.append((x, y))
        # 生成后面的边
        else:
            p1 = rd(0, len(appeared_node) - 1)
            x = appeared_node[p1]  # 第一个点从已经出现的点中选择
            p2 = rd(0, len(not_appeared_node) - 1)
            y = not_appeared_node[p2]
            appeared_node.append(y)  # 第二个点从没有出现的点中选择
            not_appeared_node = list(set(node).difference(set(appeared_node)))
            # 必须保证第一个点的排序在第二个点之前
            if node.index(y) < node.index(x):
                result.append((y, x))
            else:
                result.append((x, y))
    # 生成后m - n + 1条边
    while len(result) != m:
        p1 = rd(0, n - 2)
        p2 = rd(p1 + 1, n - 1)
        x = node[p1]
        y = node[p2]
        # 如果该条边已经生成过，则重新生成
        if (x, y) in result:
            continue
        else:
            result.append((x, y))

    return result
    # matrix = np.zeros((n, n))
    # for i in range(len(result)):
    #     matrix[result[i][0], result[i][1]] = 1
    #
    # return matrix


class Graph(object):
    def __init__(self, G):
        self.G = G
        self.color = [0] * len(G)
        self.isDAG = True

    def DFS(self, i):
        self.color[i] = 1
        for j in range(len(self.G)):
            if self.G[i][j] != 0:
                if self.color[j] == 1:
                    self.isDAG = False
                elif self.color[j] == -1:
                    continue
                else:
                    # print('We are visiting node' + str(j + 1))
                    self.DFS(j)
        self.color[i] = -1

    # 利用深度优先搜索判断一个图是否为DAG
    def DAG(self):
        for i in range(len(self.G)):
            if self.color[i] == 0:
                self.DFS(i)


edges = random_graph(6, 10)
G = nx.DiGraph()
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', linewidths=1, font_size=10)
plt.title('Random Graph Visualization')
plt.show()

# G = Graph(G)
# G.DAG()
# print(G.isDAG)
