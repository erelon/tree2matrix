import tqdm
from nltk.corpus import wordnet2021 as wn
from nltk.tokenize import wordpunct_tokenize
import networkx as nx


def article2tree(article: str):
    G = nx.DiGraph()
    for word in wordpunct_tokenize(article):
        morph = wn.morphy(word)
        if morph is None:
            continue
        path = wn.synsets(morph)[0].hypernym_paths()[0]
        for syn in path:
            syn = syn.name()
            if syn in G.nodes:
                G.nodes[syn]["count"] += 1
            else:
                G.add_node(syn)
                G.nodes[syn]["count"] = 1
        for s1, s2 in zip(path[:-1], path[1:]):
            s1 = s1.name()
            s2 = s2.name()
            G.add_edge(s1, s2)
    return G


def get_structure(Gs, prune=0):
    structure = nx.DiGraph()
    structure.add_edge("word", "entity.n.01")
    structure.add_edge("word", "verb")
    structure.nodes["verb"]["count"] = 0

    for g in tqdm.tqdm(Gs):
        structure.add_edges_from(g.edges)
        for n in g.nodes:
            if n not in structure:
                continue
            if "count" not in structure.nodes[n]:
                structure.nodes[n]["count"] = 0
            structure.nodes[n]["count"] += g.nodes[n]["count"]

    for node in structure.nodes:
        if ".v." in node and len(structure.pred[node]) == 0:
            structure.add_edge("verb", node)
            structure.nodes["verb"]["count"] += structure.nodes[node]["count"]

    structure.nodes["word"]["count"] = structure.nodes["verb"]["count"] + structure.nodes["entity.n.01"]["count"]

    for node in list(structure.nodes):
        if structure.nodes[node]["count"] <= prune:
            structure.remove_node(node)

    for node in structure.nodes:
        structure.nodes[node]["count"] = 0

    return structure
