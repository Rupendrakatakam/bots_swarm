#5 nodes a map of 100x100 left bottom edge is (0,0) - refernce point and right top edge is (100,100),  from the refence point where are the nodes and with rspect to each other nodes are located in the map using graph theory  

import matplotlib.pyplot as plt
import numpy as np 


# nodes are at (10,10), (20,20), (30,30), (40,40), (50,50)
nodes = {
    'A': (10,10),
    'B': (20,20),
    'C': (30,30),
    'D': (40,40),
    'E': (50,50)
}

x_coords = [coord[0] for coord in nodes.values()]
y_coords = [coord[1] for coord in nodes.values()]
labels = list(nodes.keys())

plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, color='red', s=100) # Plot nodes

for label, x, y in zip(labels, x_coords, y_coords):
    plt.text(x + 2, y + 2, label, fontsize=12, fontweight='bold')

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("5-Node Graph Map (100x100)")
plt.xlabel("X-axis (Distance from Reference)")
plt.ylabel("Y-axis (Distance from Reference)")

plt.show()