#5 nodes a map of 100x100 left bottom edge is (0,0) - refernce point and right top edge is (100,100),  from the refence point where are the nodes and with rspect to each other nodes are located in the map using graph theory  

import matplotlib.pyplot as plt
import numpy as np 


# nodes are at (10,10), (20,20), (30,30), (40,40), (50,50)
nodes = np.array([[10,10], [20,20], [30,30], [40,40], [50,50]])

#create a map of 100x100
grid = np.zeros((100,100))
plt.figure()
plt.imshow(grid)
plt.xticks(np.arange(0,101,10))
plt.yticks(np.arange(0,101,10))
plt.grid(True)
plt.title("100x100 Grid")
plt.show()

