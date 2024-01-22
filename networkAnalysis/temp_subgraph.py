import networkx as nx
import copy

# 创建原始图
G = nx.Graph()
G.add_edge(1, 2)
G.nodes[1]['attribute'] = 'original'

subG = G.subgraph([1, 2])
print(subG.nodes[1]['attribute'])

G.nodes[1]['attribute'] = 'modified from original graph'
print(subG.nodes[1]['attribute'])

subG.nodes[1]['attribute'] = 'modified from subgraph'
print(G.nodes[1]['attribute'])
