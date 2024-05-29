import networkx as nx
import pickle
import copy
import tqdm

from examples.text.text_tree_creator import article2tree, get_structure
from main import Tree2Matrix, MatricesDendrogram

if __name__ == '__main__':
    with open("develop.txt", "r") as f:
        raw_train_data = f.readlines()
    with open("topics.txt", "r") as f:
        topics = f.readlines()

    topics = topics[::2]
    topics = [i.strip() for i in topics]
    train_text_data = raw_train_data[2::4]
    train_topics = [i.strip(">\n").split()[2:] for i in raw_train_data[::4]]

    Gs = []
    for art in tqdm.tqdm(train_text_data):
        G = article2tree(art)
        Gs.append(G)

    structure = get_structure(Gs, 3)
    nx.write_gml(structure, "3_pruned_structure.gml")

    # Embed:
    eGs = []
    for g in tqdm.tqdm(Gs):
        embed_g = copy.deepcopy(structure)
        for n in g.nodes:
            if n in embed_g:
                embed_g.nodes[n]["count"] = g.nodes[n]["count"]
        eGs.append(embed_g)

    # Save to be safe
    with open("all_articles_as_trees.pkl", "wb") as f:
        pickle.dump(eGs, f)

    # Load if needed
    # with open("all_articles_as_trees.pkl", "rb") as f:
    #     eGs = pickle.load(f)

    M = []
    t2m = Tree2Matrix()
    for t in tqdm.tqdm(eGs):
        m, N = t2m.transform(t, "count")
        M.append(m)

    md = MatricesDendrogram()
    matrices_result, new_order_names = md.transform(M, N)

    with open("results.pkl", "wb") as f:
        pickle.dump([matrices_result, new_order_names], f)
