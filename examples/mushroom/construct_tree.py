import pickle

import pandas as pd
from scipy.cluster import hierarchy
import networkx as nx
from tqdm import tqdm

from main import Tree2Matrix, MatricesDendrogram

# Load the data from CSV
data = pd.read_csv('mushroom.csv')
# We want to find if the mushroom is edible
del data["class"]

# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data)

# Transpose the data to cluster columns instead of rows
data_transposed = data_encoded.T

# Perform hierarchical clustering
Z = hierarchy.linkage(data_transposed, "ward")

# Plot the dendrogram and retrieve the dendrogram dictionary
T = hierarchy.dendrogram(Z, labels=data_transposed.index, orientation='top', no_plot=True)

# Build a NetworkX tree from the dendrogram dictionary
G = hierarchy.to_tree(Z, rd=False)

# Convert the NetworkX tree to a NetworkX graph
G_nx = nx.DiGraph()

# Function to recursively add nodes and edges to the NetworkX graph
label2node = {}


def add_node(node):
    if node.is_leaf():
        G_nx.add_node(node.id, label=T['ivl'][node.id])
        label2node[T['ivl'][node.id]] = node.id
    else:
        G_nx.add_node(node.id)
        for child in [node.left, node.right]:
            G_nx.add_edge(node.id, child.id)
            add_node(child)


add_node(G)

M = []
t2m = Tree2Matrix()
for id, mush in tqdm(data_encoded.iterrows(), total=len(data_encoded)):
    # Create a tree for the mushroom
    G = pickle.loads(pickle.dumps(G_nx))
    for atrrib, d in mush.items():
        node_id = label2node[atrrib]
        G.nodes[node_id]["val"] = d
    m, N = t2m.transform(G, "val")
    M.append(m)
    del G

md = MatricesDendrogram()
matrices_result, new_order_names = md.transform(M, N)

with open("results.pkl", "wb") as f:
    pickle.dump([matrices_result, new_order_names], f)
