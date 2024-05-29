import pickle

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from main import Tree2Matrix, MatricesDendrogram

if __name__ == '__main__':
    data = pd.read_csv("sub_pca_tax_7_log_rare_bact_5_without_viruses.csv")
    tag = pd.read_csv("tag_for_learning.csv")["Tag"]

    del data["Unnamed: 0"]

    bact2node = {}
    structure_tree = nx.DiGraph()
    for bacteria in data.columns:
        bacteria_path = bacteria.split(";")
        for b_a, b_b in zip(bacteria_path[:-1], bacteria_path[1:]):
            if len(b_b) == 3:
                break
            structure_tree.add_edge(b_a, b_b)
            bact2node[bacteria] = b_b
    for r in [n for n in structure_tree.nodes if len(structure_tree.pred[n]) == 0]:
        structure_tree.add_edge("root", r)

    M = []
    t2m = Tree2Matrix()
    for id, person in tqdm(data.iterrows(), total=len(data)):
        G = pickle.loads(pickle.dumps(structure_tree))
        for bact, val in person.items():

            if bact in bact2node:
                G.nodes[bact2node[bact]]["val"] = val

        m, N = t2m.transform(G, "val")
        M.append(m)
        del G

    md = MatricesDendrogram()
    matrices_result, new_order_names = md.transform(M, N)

    with open("results.pkl", "wb") as f:
        pickle.dump([matrices_result, new_order_names], f)
