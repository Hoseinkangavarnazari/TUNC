import random
import time

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.max_num_edges = num_vertices * (num_vertices - 1) // 2
        self.graph = [[] for _ in range(num_vertices)]

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def generate_random_edges(self, num_edges):
        if num_edges > self.max_num_edges:
            raise ValueError(f"Number of edges exceeds the maximum possible ({self.max_num_edges}) for the given number of num_vertices.")

        count_edges = 0
        while count_edges < num_edges:
            u, v = random.randint(0, self.num_vertices - 1), random.randint(0, self.num_vertices - 1)
            if u != v and v not in self.graph[u]:
                self.add_edge(u, v)
                count_edges += 1

    def visualize_graph(self, coloring_result, no_figure):
        # Create a NetworkX graph from the adjacency list
        plt.figure(no_figure)
        G = nx.Graph()
        for vertex in range(self.num_vertices):
            G.add_node(vertex)
            for neighbor in self.graph[vertex]:
                G.add_edge(vertex, neighbor)

        # Define a color map based on the coloring result
        color_map = [coloring_result[node] for node in G.nodes()]

        pos = nx.kamada_kawai_layout(G)

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=600, cmap=plt.cm.Set3)

    def color_greedy(self):
        # Initialize color array
        result = [-1] * self.num_vertices

        # Assign colors to the num_vertices
        for vertex in range(self.num_vertices):
            available_colors = set(range(self.num_vertices))
            for neighbor in self.graph[vertex]:
                if result[neighbor] != -1:
                    available_colors.discard(result[neighbor])

            # Assign the smallest available color
            result[vertex] = min(available_colors)

        # Calculate the number of colors used
        num_colors = max(result) + 1

        return result, num_colors

    def color_welsh_powell(self):
        # Sort num_vertices by degree in descending order
        sorted_vertices = sorted(range(self.num_vertices), key=lambda x: len(self.graph[x]), reverse=True)

        # Initialize color array
        result = [-1] * self.num_vertices

        # Assign the first color to the first vertex
        result[sorted_vertices[0]] = 0

        # Assign colors to the remaining num_vertices
        for i in range(1, self.num_vertices):
            available_colors = [True] * self.num_vertices
            for neighbor in self.graph[sorted_vertices[i]]:
                if result[neighbor] != -1:
                    available_colors[result[neighbor]] = False

            # Find the first available color
            color = 0
            while not available_colors[color]:
                color += 1

            result[sorted_vertices[i]] = color

        # Calculate the number of colors used
        num_colors = max(result) + 1

        return result, num_colors


def run_sample():
    g = Graph(6)
    g.generate_random_edges(10)

    print("Generated Graph:")
    for i in range(g.num_vertices):
        print(f"Vertex {i}: {g.graph[i]}")

    greedy_coloring_result, greedy_num_colors = g.color_greedy()
    welsh_powell_coloring_result, welsh_powell_num_colors = g.color_welsh_powell()

    print("\nColoring Result:")
    print("Vertex \tColor(Greedy) \tColor(Welsh-Powell)")
    for i in range(len(greedy_coloring_result)):
        print(f"{i}\t{greedy_coloring_result[i]}\t\t{welsh_powell_coloring_result[i]}")
    print(f"\nNumber of colors used (Greedy): {greedy_num_colors}")
    print(f"Number of colors used (Welsh-Powell): {welsh_powell_num_colors}")

    g.visualize_graph(greedy_coloring_result, 1)
    g.visualize_graph(welsh_powell_coloring_result, 2)
    plt.show()


def simulate(num_nodes, num_sims, step_edges):
    columns = ['Simulation No.', 'Edges', 'Greedy Colors', 'Welsh-Powell Colors']
    df = pd.DataFrame(columns=columns)

    range_edges = range(5, Graph(num_nodes).max_num_edges + 1, step_edges)

    for num_edges in range_edges:
        for sim in range(1, num_sims + 1):
            g = Graph(num_nodes)
            g.generate_random_edges(num_edges)

            _, greedy_num_colors = g.color_greedy()
            _, welsh_powell_num_colors = g.color_welsh_powell()

            new_row = pd.DataFrame([[sim, num_edges, greedy_num_colors, welsh_powell_num_colors]], columns=columns)
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(f"./graphColoring/sim_{num_nodes}nodes_{num_sims}sims.csv", sep='\t', index=False)


def plot_results(num_nodes, num_sims):
    df = pd.read_csv(f"./graphColoring/sim_{num_nodes}nodes_{num_sims}sims.csv", sep='\t')

    df['Greedy Colors'] = pd.to_numeric(df['Greedy Colors'], errors='coerce')
    df['Welsh-Powell Colors'] = pd.to_numeric(df['Welsh-Powell Colors'], errors='coerce')
    df['Edges'] = pd.to_numeric(df['Edges'], errors='coerce')

    greedy_data = df[['Edges', 'Greedy Colors']]
    greedy_data = greedy_data.rename(columns={'Greedy Colors': 'Colors'})
    greedy_data['Algorithm'] = 'Greedy'

    welsh_powell_data = df[['Edges', 'Welsh-Powell Colors']]
    welsh_powell_data = welsh_powell_data.rename(columns={'Welsh-Powell Colors': 'Colors'})
    welsh_powell_data['Algorithm'] = 'Welsh-Powell'

    combined_data = pd.concat([greedy_data, welsh_powell_data])

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x='Edges', y='Colors', hue='Algorithm', data=combined_data, palette='Set2')

    # 计算每种算法的平均颜色数并绘制
    for algorithm in ['Greedy', 'Welsh-Powell']:
        mean_colors = combined_data[combined_data['Algorithm'] == algorithm].groupby('Edges')['Colors'].mean().reset_index()
        mean_colors['Edges'] = mean_colors['Edges'].astype(str)
        plt.scatter(mean_colors['Edges'], mean_colors['Colors'], alpha=1, label=f'{algorithm} Mean')
        plt.plot(mean_colors['Edges'], mean_colors['Colors'], linestyle='--' if algorithm == 'Welsh-Powell' else '-')

    plt.title(f'Coloring Algorithm Comparison ({num_nodes} Nodes, {num_sims} Simulations)')
    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Colors Used')
    plt.legend()
    # plt.savefig(f"./graphColoring/sim_{num_nodes}nodes_{num_sims}sims.eps", format='eps', dpi=1000)
    plt.savefig(f"./graphColoring/sim_{num_nodes}nodes_{num_sims}sims.png", format='png', dpi=1000)
    plt.close()


if __name__ == "__main__":
    # run_sample()

    num_simulations = 5000
    steps = 5

    for nodes in [10, 20, 30]:
        if nodes == 10:
            steps = 5
        elif nodes == 20:
            steps = 15
        elif nodes == 30:
            steps = 30

        start_time = time.time()
        simulate(nodes, num_simulations, steps)
        elapsed_time = time.time() - start_time
        print(f"Time taken for {nodes} nodes ({num_simulations} runs): {elapsed_time:.3f} seconds")

    for nodes in [10, 20, 30]:
        plot_results(nodes, num_simulations)
