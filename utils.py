import networkx as nx
import random


def generate_random_tree(depth):
    G = nx.DiGraph()
    G.add_node(0, val=random.randint(1, 100), name="root")  # Root node with random value
    nodes_at_depth = [0]  # Nodes at the current depth
    for d in range(1, depth + 1):
        next_nodes_at_depth = []
        for node in nodes_at_depth:
            num_children = random.randint(0, 3)  # Random number of children per node (including 0)
            children = range(max(G.nodes) + 1, max(G.nodes) + num_children + 1)
            G.add_nodes_from(children)
            for child in children:
                G.add_edge(node, child)
                v = random.randint(1, 100)  # Assign a random value to the child node
                G.nodes[child]['val'] = v
                G.nodes[child]['name'] = str(v)
            next_nodes_at_depth.extend(children)
        nodes_at_depth = next_nodes_at_depth
    return G


def rand_tree_values(G):

    for n in G.nodes:
        v = random.randint(1, 100)  # Assign a random value to the child node
        G.nodes[n]['val'] = v
        G.nodes[n]['name'] = str(v)
    return G

def get_root(G):
    roots = [n for n in G.nodes if len(G.pred[n]) == 0]
    if len(roots) != 1:
        raise Exception("Not a tree")
    return roots[0]
