import random

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[] for _ in range(vertices)]

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def greedy_coloring(self):
        # Initialize color array
        result = [-1] * self.vertices

        # Assign colors to the vertices
        for vertex in range(self.vertices):
            available_colors = set(range(self.vertices))
            for neighbor in self.graph[vertex]:
                if result[neighbor] != -1:
                    available_colors.discard(result[neighbor])

            # Assign the smallest available color
            result[vertex] = min(available_colors)

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
    g.generate_random_graph(8)  # Generate a random graph with 8 edges

    # Print the generated graph
    print("Generated Graph:")
    for i in range(g.vertices):
        print(f"Vertex {i}: {g.graph[i]}")

    # Get the result of graph coloring using the greedy algorithm
    coloring_result, num_colors = g.greedy_coloring()

    print("\nGraph Coloring:")
    print("Vertex \tColor")
    for i in range(len(coloring_result)):
        print(f"{i}\t{coloring_result[i]}")

    print(f"\nNumber of colors used: {num_colors}")
