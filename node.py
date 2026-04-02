#5 nodes a map of 100x100 left bottom edge is (0,0) - refernce point and right top edge is (100,100),  from the refence point where are the nodes and with rspect to each other nodes are located in the map using graph theory  

# import matplotlib.pyplot as plt
# import numpy as np 
# import math


# # nodes are at (10,10), (20,20), (30,30), (40,40), (50,50)
# nodes = {
#     'Ref': (0,0),
#     'A': (10,10),
#     'B': (20,20),
#     'C': (30,30),
#     'D': (40,40),
#     'E': (50,50)
# }

# # 1. Calculate Distances from Reference Point (0,0)
# print("--- Distances from Reference Point (0,0) ---")
# for name, coords in nodes.items():
#     if name != 'Ref':
#         # math.dist calculates the Euclidean distance between two points
#         dist = math.dist(nodes['Ref'], coords)
#         print(f"To Node {name}: {dist:.2f} units")

# print("\n--- Distance Matrix (Node to Node) ---")
# # 2. Calculate Pairwise Distances between nodes
# node_names = ['A', 'B', 'C', 'D', 'E']

# # Print a simple header for our table
# print(f"{'':>4} | " + " | ".join([f"{n:>5}" for n in node_names]))
# print("-" * 45)

# # Calculate and print the distance from every node to every other node
# for n1 in node_names:
#     row_data = [f"{n1:>4} |"]
#     for n2 in node_names:
#         # Distance from node n1 to node n2
#         dist = math.dist(nodes[n1], nodes[n2])
#         row_data.append(f"{dist:>5.1f}")
    
#     print(" | ".join(row_data))

# #visualization
# x_coords = [coord[0] for coord in nodes.values()]
# y_coords = [coord[1] for coord in nodes.values()]
# labels = list(nodes.keys())

# plt.figure(figsize=(8, 8))
# plt.scatter(x_coords, y_coords, color='red', s=100) # Plot nodes

# for label, x, y in zip(labels, x_coords, y_coords):
#     plt.text(x + 2, y + 2, label, fontsize=12, fontweight='bold')

# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.title("5-Node Graph Map (100x100)")
# plt.xlabel("X-axis (Distance from Reference)")
# plt.ylabel("Y-axis (Distance from Reference)")

# plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import math

# 1. Define nodes and their coordinates (including the Reference point)
positions = {
    'Ref': (0, 0),
    'A': (10, 10),
    'B': (20, 80),
    'C': (50, 50),
    'D': (80, 20),
    'E': (90, 90)
}

# 2. Initialize the Graph
G = nx.Graph()

# Add nodes to the graph and attach their (x, y) coordinates as an attribute
for node, pos in positions.items():
    G.add_node(node, pos=pos)

# 3. Add Edges and calculate Weights (Distances)
# We will create a "complete graph" where every node connects to every other node
nodes_list = list(G.nodes())
for i in range(len(nodes_list)):
    for j in range(i + 1, len(nodes_list)):
        node1 = nodes_list[i]
        node2 = nodes_list[j]
        
        # Calculate Euclidean distance
        dist = math.dist(positions[node1], positions[node2])
        
        # Add the edge and store the distance as the 'weight'
        G.add_edge(node1, node2, weight=dist)

# ---------------------------------------------------------
# Extracting and Printing the Data from the NetworkX Graph
# ---------------------------------------------------------

print("--- Distances from Reference Point (0,0) ---")
# G.neighbors('Ref') looks at all nodes connected to 'Ref'
for neighbor in G.neighbors('Ref'):
    # We access the weight dictionary of the edge between 'Ref' and the neighbor
    distance = G['Ref'][neighbor]['weight']
    print(f"To Node {neighbor}: {distance:.2f} units")


print("\n--- Distance Matrix (Node to Node) ---")
main_nodes = ['A', 'B', 'C', 'D', 'E']

# Print header
print(f"{'':>4} | " + " | ".join([f"{n:>5}" for n in main_nodes]))
print("-" * 45)

# Print rows by looking up the edge weights in the graph
for n1 in main_nodes:
    row = [f"{n1:>4} |"]
    for n2 in main_nodes:
        if n1 == n2:
            row.append(f"{0.0:>5.1f}") # Distance to itself is 0
        else:
            # Retrieve the weight we saved earlier
            dist = G[n1][n2]['weight']
            row.append(f"{dist:>5.1f}")
    print(" | ".join(row))

# ---------------------------------------------------------
# Visualizing the Graph
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))

# Draw the nodes using the saved 'pos' attribute
nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, positions, font_weight='bold')

# Draw the edges. Let's make connections to the Reference point red and dashed, 
# and all other connections light gray to keep the map readable.
ref_edges = [(u, v) for u, v in G.edges() if u == 'Ref' or v == 'Ref']
other_edges = [(u, v) for u, v in G.edges() if u != 'Ref' and v != 'Ref']

nx.draw_networkx_edges(G, positions, edgelist=ref_edges, edge_color='red', style='dashed')
nx.draw_networkx_edges(G, positions, edgelist=other_edges, edge_color='gray', alpha=0.3)

plt.title("NetworkX Graph: Distances Stored as Edge Weights")
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('on') # Turn on the axes to see the 0 to 100 scale
plt.xlim(-5, 100)
plt.ylim(-5, 100)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()