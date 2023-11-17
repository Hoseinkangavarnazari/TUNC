import random

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[] for _ in range(vertices)]

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def welsh_powell(self):
        # Sort vertices by degree in descending order
        sorted_vertices = sorted(range(self.vertices), key=lambda x: len(self.graph[x]), reverse=True)

        # Initialize color array
        result = [-1] * self.vertices

        # Assign the first color to the first vertex
        result[sorted_vertices[0]] = 0

        # Assign colors to the remaining vertices
        for i in range(1, self.vertices):
            available_colors = [True] * self.vertices
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

    def generate_random_graph(self, num_edges):
        # Generate a random graph with a specified number of edges
        if num_edges > self.vertices * (self.vertices - 1) // 2:
            raise ValueError("Number of edges exceeds the maximum possible for the given number of vertices.")

        edge_count = 0
        while edge_count < num_edges:
            u, v = random.randint(0, self.vertices - 1), random.randint(0, self.vertices - 1)
            if u != v and v not in self.graph[u]:
                self.add_edge(u, v)
                edge_count += 1

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    g = Graph(6)
    g.generate_random_graph(16)  # Generate a random graph with 8 edges

    # Print the generated graph
    print("Generated Graph:")
    for i in range(g.vertices):
        print(f"Vertex {i}: {g.graph[i]}")

    # Get the result of graph coloring using Welsh-Powell algorithm
    coloring_result, num_colors = g.welsh_powell()

    print("\nGraph Coloring:")
    print("Vertex \tColor")
    for i in range(len(coloring_result)):
        print(f"{i}\t{coloring_result[i]}")

    print(f"\nNumber of colors used: {num_colors}")
