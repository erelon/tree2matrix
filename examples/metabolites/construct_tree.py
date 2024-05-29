import copy
import pickle

import pandas as pd
import networkx as nx
import numpy as np
import tqdm

from main import Tree2Matrix, MatricesDendrogram


def structure_tree_builder(ref: pd.DataFrame):
    G = nx.DiGraph()
    structure_dict = {}
    for entry in ref.iterrows():
        entry = entry[1]
        levels = [entry["Parent Level 6"], entry["Parent Level 5"], entry["Parent Level 4"], entry["Parent Level 3"],
                  entry["Parent Level 2"], entry["Parent Level 1"], entry["Subclass"], entry["Class"],
                  entry["Superclass"], entry["Kingdom"]]
        levels.reverse()
        k = 0
        for l1, l2 in zip(levels[:-1], levels[1:]):
            if type(l2) != str and np.isnan(l2):
                if entry["molecular_formula"] not in structure_dict and k != 0:
                    structure_dict[entry["molecular_formula"]] = f"{k}_{l1}"
                continue
            G.add_edge(f"{k}_{l1}", f"{k + 1}_{l2}")
            k += 1

    roots = [n for n in G if len(G.pred[n]) == 0]
    for r in roots:
        G.add_edge("matter", r)
    for n in G.nodes:
        if "count" not in G.nodes[n]:
            G.nodes[n]["count"] = 0.
    return G, structure_dict


if __name__ == '__main__':
    metab_with_formula = pd.read_csv("metab_with_formula.csv")
    pub_table = pd.read_csv("pub_table.csv")
    classyfier = pd.read_csv("classyfire_20220519130434.csv")

    key2formula = pub_table[["inchikey", "molecular_formula"]]
    classyfier = classyfier.rename(columns={"InChIKey": "inchikey"})
    hierc = pd.merge(right=classyfier, left=key2formula, on="inchikey").drop_duplicates()
    structure_tree, structure_dict = structure_tree_builder(hierc)

    formula = metab_with_formula["formula"]
    metab = metab_with_formula.rename(index=formula)
    del metab["formula"]

    Gs = []
    for entry in metab:
        entry = metab[entry]
        G = copy.deepcopy(structure_tree)
        for k, v in entry.items():
            if k in structure_dict:
                G.nodes[structure_dict[k]]["count"] = v
        Gs.append(G)

    M = []
    t2m = Tree2Matrix()
    for t in tqdm.tqdm(Gs):
        m, N = t2m.transform(t, "count")
        M.append(m)

    md = MatricesDendrogram()
    matrices_result, new_order_names = md.transform(M, N)

    with open("results.pkl", "wb") as f:
        pickle.dump([matrices_result, new_order_names], f)
