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



'''
  Plot a graph (just a 121 node) from a DF and pass some variables
  as edge properties.
  Plot them all
'''

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


this_file = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/data/Walmart(M5)/potential_cannibals/FOODS/FOODS_1/pickle/CA_1.pickle'
df = pd.read_pickle(this_file)

idx = 0
df.iloc[idx]


# Directed graph
DG = nx.DiGraph()

victim_node = df.index[idx]
d = dict()
d['average_sales'] = -1
d['count_kmeans'] = [1,2,4]
DG.add_node(victim_node, **d)

cannibal_node = df.iloc[idx]['cannibal']
d = dict()
d['average_sales'] = -1
d['count_kmeans'] = [1,2,11]
DG.add_node(cannibal_node, **d)


vars_edges = ['MI_y1', 'MI_y0', 'n_1(days_overlap)', 
'mu_y1', 'mu_y0']
edge_properties = df.iloc[idx][vars_edges].to_dict()
edge_label = '\n'.join([f'{k}: {v:3.2f}' for k,v in edge_properties.items()])
DG.add_edge(cannibal_node, victim_node, **edge_properties, label=edge_label)


pos = nx.spring_layout(DG)
nx.draw_networkx(DG, pos, 
  with_labels=True, 
  font_size=10,
  edge_color='r')


labels = nx.get_edge_attributes(DG, 'label')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels, font_size=8)
plt.show()






## Cool visualisation
fig= plt.figure(figsize=(22,11))
pos = nx.nx_agraph.graphviz_layout(DG, prog="sfdp")
#pos = nx.nx_agraph.graphviz_layout(DG, prog="neato")
nx.draw(DG, pos, \
    node_color='lightblue', linewidths=0.5, font_size=10, \
    font_weight='bold', with_labels=False, edge_color='r')

labels = nx.get_edge_attributes(DG, 'label')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels, font_size=8)

for k, v in pos.items():
    plt.text(v[0],v[1]+18, s=k, horizontalalignment='center')

plt.show()







# To use different fonts

import matplotlib.pyplot as plt
import networkx as nx

font_names = ['Sawasdee', 'Gentium Book Basic', 'FreeMono', ]
family_names = ['sans-serif', 'serif', 'fantasy', 'monospace']


# Make a graph
G  = nx.generators.florentine_families_graph()

# need some positions for the nodes, so lay it out
pos = nx.spring_layout(G)

# create some maps for some subgraphs (not elegant way)
subgraph_members  = [G.nodes()[i:i+3] for i in xrange(0, len(G.nodes()), 3)]

plt.figure(1)
nx.draw_networkx_nodes(G, pos)


for i, nodes in enumerate(subgraph_members):
    f = font_names[(i % 3)]
    #f = family_names[(i % 4)]
    # extract the subgraph
    g = G.subgraph(subgraph_members[i])
    # draw on the labels with different fonts
    nx.draw_networkx_labels(g, pos, font_family=f, font_size=40)

# show the edges too
nx.draw_networkx_edges(G, pos)


plt.show()



#######
# Some layouts
'''

dot - “hierarchical” or layered drawings of directed graphs. This is the default tool to use if edges have directionality.

neato - “spring model” layouts. This is the default tool to use if the graph is not too large (about 100 nodes) and you don't know anything else about it. Neato attempts to minimize a global energy function, which is equivalent to statistical multi-dimensional scaling.

fdp - “spring model” layouts similar to those of neato, but does this by reducing forces rather than working with energy.

sfdp - multiscale version of fdp for the layout of large graphs.

twopi - radial layouts, after Graham Wills 97. Nodes are placed on concentric circles depending their distance from a given root node.

circo - circular layout, after Six and Tollis 99, Kauffman and Wiese 02. This is suitable for certain diagrams of multiple cyclic structures, such as certain telecommunications networks.

'''
pos = nx.nx_agraph.graphviz_layout(DG, prog="sfdp")
pos = nx.nx_agraph.graphviz_layout(DG, prog="neato")
pos = nx.nx_agraph.graphviz_layout(DG, prog="neato")
pos = nx.nx_agraph.graphviz_layout(DG, prog="twopi")




# To specifically set the position of one of the above algos as a node attribute
DG_gephi = DG.copy()
for n, p in pos.items():
    DG_gephi.nodes[n]['X'] = p[0]
    DG_gephi.nodes[n]['Y'] = p[1]
    
# Export the graph so we can visualise it with Gephi
graph_file = os.path.join(gephiFolder, 'graph_for_Gephi_3.gexf')
nx.write_gexf(DG_gephi, graph_file)
print(f'GEFX saved to {graph_file}')