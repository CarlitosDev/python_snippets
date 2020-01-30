import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



# Example: add node with some attributes
DG = nx.DiGraph()

left_node = 'activity_' + str(524)
d = dict()
d['times_played'] = 86
d['times_completed'] = 34
d['activity_type'] = 'song'
DG.add_node(left_node, **d)


# Print node info
print(nx.info(DG))



## Access info
# a.1 - one attribute from the nodes
attribute = 'activity_type'
s = nx.get_node_attributes(DG, attribute)
# a.2 - All attributes from all the nodes
all_attributes = {n: d for n, d in DG.nodes.items()}
# b - from the edges
a = DG.get_edge_data(left_node, right_node)







# Directed edge. Increase weight if exists
def update_edge(DG: 'directed graph', left_node, right_node):
  if DG.has_edge(left_node, right_node):
    DG.edges[(left_node, right_node)]['weight'] += 1
  else:
    DG.add_edge(left_node, right_node, weight=1)
  return DG

'''
# Add node if it does not exist
node_B = 'activity_' + '875'
if not DG.has_node(node_B):
  DG.add_node(node_B)
'''


# Create a directed graph
DG = nx.DiGraph()


# Add node
node_A = 'activity_' + '874'
DG.add_node(node_A)

# Add another node
node_B = 'activity_' + '865'
DG.add_node(node_B)


# Add another node
node_C = 'activity_' + '875'
DG.add_node(node_C)



print(DG.nodes)



DG = update_edge(DG, node_A, node_B)
DG = update_edge(DG, node_A, node_C)
DG = update_edge(DG, node_A, node_B)
print(DG.edges)


# Scatter the nodes
pos = nx.spring_layout(DG)

#nx.draw(DG)
nx.draw_networkx(DG, pos, with_labels=True, font_weight='bold')


# Get the edges properties
labels = nx.get_edge_attributes(DG, 'weight')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)
plt.show()









nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)



plt.show()


pos=nx.get_node_attributes(DG,'pos')
nx.draw(DG,pos)
labels = nx.get_edge_attributes(DG,'weight')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)

nx.draw_networkx_edge_labels(DG, edge_labels=labels)



DG.edges[(node_A, node_B)]['weight']

###
pos=nx.spring_layout(DG)
nx.draw(DG,pos)


nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)
plt.show()





# Add edge: user goes from 874 to 865
# this doesn't work
e_AB = (node_A, node_B, {'weight': 1})
G.add_edge(e_AB)


G.add_edges_from([(1, 2, {'color':'blue'}), (2, 3, {'weight':8})])

# this does
e_AB = (node_A, node_B)
G.add_edge(*e_AB)

# works
G.add_edge(node_A, node_B, weight=1)

G.number_of_nodes()
G.number_of_edges()


DG = nx.DiGraph()
# dict-of-dict-of-attribute
adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
e = [(u, v, {'weight': d}) for u, nbrs in adj.items()
	for v, d in nbrs.items()]
DG.update(edges=e, nodes=adj)



# Check if a node exists
#   G.has_node(node_B)
#   also: node_B in G
# Check if edge exists
#   G.has_edge(edge_AB)
# Update properties of edges
#   G.get_edge_data('a', 'b', default=0)



DG = nx.DiGraph()

# Add node
node_A = 'activity_' + '874'
DG.add_node(node_A)


# Add node if it does not exist
node_B = 'activity_' + '875'
if not DG.has_node(node_B):
  DG.add_node(node_B)






left_node = node_A
right_node = node_B

DG.edges()
DG.nodes()

G.get_edge_data(edge_AB, 'weight')
G.get_edge_data(edge_AB, 'weight')

G.get_edge_data(*edge_AB)

nx.set_edge_attributes(DG, edge_AB, 'betweenness')
edge_AB = (left_node, right_node)
DG.edges.data(edge_AB, 'weight')

DG.edges.data('weight', default=0)

DG.edges[0, 1]['labels']

DG.edges(edge_AB[0],edge_AB[1]) 

DG.edges(edge_AB, 'weight')


DG.edges.data('weight')