import random
import time

import networkx as nx
import matplotlib.pyplot as plt


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


def run_simulation_and_record():
    num_vertices_options = [10, 20, 30]
    num_simulations = 2000
    step = 5

    with open("./graphColoring/coloring_results.txt", "w") as file:
        # Writing the header with tab separation
        file.write("Vertices\tSimulation\tEdges\tGreedy Colors\tWelsh-Powell Colors\n")

        for num_vertices in num_vertices_options:
            edges_range = range(5, Graph(num_vertices).max_num_edges + 1, step)
            avg_greedy = []
            avg_welsh_powell = []

            for num_edges in edges_range:
                results_greedy = []
                results_welsh_powell = []

                for sim in range(1, num_simulations + 1):
                    g = Graph(num_vertices)
                    g.generate_random_edges(num_edges)

                    _, greedy_num_colors = g.color_greedy()
                    _, welsh_powell_num_colors = g.color_welsh_powell()

                    results_greedy.append(greedy_num_colors)
                    results_welsh_powell.append(welsh_powell_num_colors)

                    # Writing each simulation result
                    file.write(f"{num_vertices}\t{sim}\t{num_edges}\t{greedy_num_colors}\t{welsh_powell_num_colors}\n")

                avg_greedy.append(sum(results_greedy) / num_simulations)
                avg_welsh_powell.append(sum(results_welsh_powell) / num_simulations)

                file.write(f"Average\t{num_vertices}\t{num_edges}\t{avg_greedy[-1]:.2f}\t{avg_welsh_powell[-1]:.2f}\n")

        # file.write(f"Total Time Used for {len(num_vertices_options)} groups of nodes with {num_simulations} simulations: {time.time() - start_time:.2f} seconds\n")


def plot_simulation_results():
    x_10 = []
    y_greedy_10 = []
    y_welsh_powell_10 = []

    x_10_avg = []
    y_greedy_10_avg = []
    y_welsh_powell_10_avg = []

    x_20 = []
    y_greedy_20 = []
    y_welsh_powell_20 = []

    x_20_avg = []
    y_greedy_20_avg = []
    y_welsh_powell_20_avg = []

    x_30 = []
    y_greedy_30 = []
    y_welsh_powell_30 = []

    x_30_avg = []
    y_greedy_30_avg = []
    y_welsh_powell_30_avg = []

    with open("./graphColoring/coloring_results.txt", "r") as file:
        next(file)  # Skip the header
        for line in file:
            parts = line.strip().split('\t')
            if parts[0] != "Average":
                vertices, _, edges, greedy_colors, welsh_powell_colors = parts
                if vertices == '10':
                    x_10.append(int(edges))
                    y_greedy_10.append(int(greedy_colors))
                    y_welsh_powell_10.append(int(welsh_powell_colors))
                elif vertices == '20':
                    x_20.append(int(edges))
                    y_greedy_20.append(int(greedy_colors))
                    y_welsh_powell_20.append(int(welsh_powell_colors))
                elif vertices == '30':
                    x_30.append(int(edges))
                    y_greedy_30.append(int(greedy_colors))
                    y_welsh_powell_30.append(int(welsh_powell_colors))
            else:
                _, vertices, edges, avg_greedy, avg_welsh_powell = parts
                if vertices == '10':
                    x_10_avg.append(int(edges))
                    y_greedy_10_avg.append(float(avg_greedy))
                    y_welsh_powell_10_avg.append(float(avg_welsh_powell))
                elif vertices == '20':
                    x_20_avg.append(int(edges))
                    y_greedy_20_avg.append(float(avg_greedy))
                    y_welsh_powell_20_avg.append(float(avg_welsh_powell))
                elif vertices == '30':
                    x_30_avg.append(int(edges))
                    y_greedy_30_avg.append(float(avg_greedy))
                    y_welsh_powell_30_avg.append(float(avg_welsh_powell))

    plt.figure(figsize=(30, 14))

    plt.scatter(x_10, y_greedy_10, color='blue', alpha=0.05, label='Greedy (10 Vertices)')
    plt.scatter(x_10, y_welsh_powell_10, color='red', alpha=0.05, label='Welsh-Powell (10 Vertices)')
    plt.plot(x_10_avg, y_greedy_10_avg, color='blue', label='Greedy (avg) - 10 Vertices')
    plt.plot(x_10_avg, y_welsh_powell_10_avg, color='red', label='Welsh-Powell (avg) - 10 Vertices')

    plt.scatter(x_20, y_greedy_20, color='green', alpha=0.05, label='Greedy (20 Vertices)')
    plt.scatter(x_20, y_welsh_powell_20, color='orange', alpha=0.05, label='Welsh-Powell (20 Vertices)')
    plt.plot(x_20_avg, y_greedy_20_avg, color='green', label='Greedy (avg) - 20 Vertices')
    plt.plot(x_20_avg, y_welsh_powell_20_avg, color='orange', label='Welsh-Powell (avg) - 20 Vertices')

    plt.scatter(x_30, y_greedy_30, color='purple', alpha=0.05, label='Greedy (30 Vertices)')
    plt.scatter(x_30, y_welsh_powell_30, color='brown', alpha=0.05, label='Welsh-Powell (30 Vertices)')
    plt.plot(x_30_avg, y_greedy_30_avg, color='purple', label='Greedy (avg) - 30 Vertices')
    plt.plot(x_30_avg, y_welsh_powell_30_avg, color='brown', label='Welsh-Powell (avg) - 30 Vertices')

    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Colors Used')
    plt.title('Graph Coloring')
    plt.legend()
    plt.grid(True)
    plt.savefig('./graphColoring/coloring_results.png')


if __name__ == "__main__":
    # run_sample()
    run_simulation_and_record()
    plot_simulation_results()
