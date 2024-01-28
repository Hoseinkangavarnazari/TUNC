import networkx as nx
import matplotlib.font_manager

import copy

def subgraph():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.nodes[1]['attribute'] = 'original'

    subG = G.subgraph([1, 2])
    print(subG.nodes[1]['attribute'])

    G.nodes[1]['attribute'] = 'modified from original graph'
    print(subG.nodes[1]['attribute'])

    subG.nodes[1]['attribute'] = 'modified from subgraph'
    print(G.nodes[1]['attribute'])


def lsfonts():
    fonts = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for font in sorted(fonts):
        print(font)

if __name__ == '__main__':
    lsfonts()