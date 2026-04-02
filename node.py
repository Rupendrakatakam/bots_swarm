#5 nodes a map of 100x100 left bottom edge is (0,0) - refernce point and right top edge is (100,100),  from the refence point where are the nodes and with rspect to each other nodes are located in the map using graph theory  

import networkx as nx
import matplotlib.pyplot as plt
import math

# 1. Define nodes and their coordinates
node_coords = {
    'A': (10, 10),
    'B': (20, 80),
    'C': (50, 50),
    'D': (80, 20),
    'E': (90, 90)
}

# 2. Initialize an empty Graph
G = nx.Graph()

# 3. Add nodes to the graph along with their (x, y) positions
for node, pos in node_coords.items():
    G.add_node(node, pos=pos)

# 4. Connect nodes based on a distance rule
# Let's say we only connect nodes that are less than 60 units apart
threshold = 60.0
node_names = list(node_coords.keys())

# Loop through every unique pair of nodes
for i in range(len(node_names)):
    for j in range(i + 1, len(node_names)):
        n1 = node_names[i]
        n2 = node_names[j]
        
        # Calculate Euclidean distance
        dist = math.dist(node_coords[n1], node_coords[n2])
        
        # If they are close enough, add an edge!
        if dist <= threshold:
            # We save the distance as the edge 'weight'
            G.add_edge(n1, n2, weight=round(dist, 1))

# 5. Visualization Setup
plt.figure(figsize=(8, 8))

# Extract the positions we saved earlier so NetworkX knows where to draw them
pos = nx.get_node_attributes(G, 'pos')

# Draw the nodes and the connecting lines (edges)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
        node_size=1000, font_size=12, font_weight='bold', edge_color='gray')

# Draw the distance numbers on top of the lines
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Set up the 100x100 map boundaries
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Nodes Connected if Distance ≤ {threshold}")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# NetworkX hides axes by default, so we turn them back on to see the 100x100 grid
plt.axis('on') 
plt.show()