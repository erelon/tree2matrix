from collections import defaultdict
import random
import igraph
import numpy as np
import pandas as pd
from plotly.figure_factory._dendrogram import sch
import networkx as nx
from textreeCreate import create_tax_tree


# from sonLib.nxtree import
def seperate_according_to_tag(otu, tag):
    merge = otu.join(tag, on=otu.index).dropna()
    otu_1 = merge.loc[merge[merge.columns[-1]] == 1]
    del otu_1[merge.columns[-1]]
    otu_0 = merge.loc[merge[merge.columns[-1]] == 0]
    del otu_0[merge.columns[-1]]
    return otu_0, otu_1


def save_2d(otu, name, path):
    otu.dump(f"{path}/{name}.npy")


def get_map_ieee(tree: igraph.Graph, permute=-1):
    # self.set_height()
    # self.set_width()
    order, layers, _ = tree.bfs(0)
    # the biggest layer len of the phylogenetic tree
    width = max(layers[-1] - layers[-2], layers[-2] - layers[-3])
    # number of layers in phylogenetic tree
    height = len(layers)
    m = np.zeros((height, width))

    layers_ = defaultdict(list)
    k = 0
    for index, i in enumerate(order):
        if index != layers[k]:
            layers_[k].append(i)
        else:
            k += 1
            layers_[k].append(i)

    i = 0
    for l in layers_:
        j = 0
        for n in layers_[l]:
            m[i][j] = tree.vs.find(n)["_nx_name"][1]
            j = j + 1
        i += 1
    return np.delete(m, height - 1, 0)


def find_root(G, child):
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        print(f"found root: {child}")
        return child
    else:
        return find_root(G, parent[0])


def dfs_rec(tree, node, added, depth, m, N):
    added[node] = True
    neighbors = tree.neighbors(node)
    sum = 0
    num_of_sons = 0
    num_of_descendants = 0
    for neighbor in neighbors:
        if added[neighbor] == True:
            continue
        val, m, descendants, N, name = dfs_rec(tree, neighbor, added, depth + 1, m, N)
        sum += val
        num_of_sons += 1
        num_of_descendants += descendants

    if num_of_sons == 0:
        value = tree.vs[node]["_nx_name"][1]  # the value
        n = np.zeros((len(m), 1)) / 0
        n[depth, 0] = value
        m = np.append(m, n, axis=1)

        name = ";".join(tree.vs[node]["_nx_name"][0])
        n = np.empty((len(m), 1), dtype=f"<U{len(name)}")
        n[depth, 0] = name
        N = np.append(N, n, axis=1)
        return value, m, 1, N, name

    avg = sum / num_of_sons
    name = ";".join(name.split(";")[:-1])
    for j in range(num_of_descendants):
        m[depth][len(m.T) - 1 - j] = avg
        N[depth][len(m.T) - 1 - j] = name

    return avg, m, num_of_descendants, N, name


def dfs_(tree, m, N):
    nv = tree.vcount()
    added = [False for v in range(nv)]
    _, m, _, N, _ = dfs_rec(tree, 0, added, 0, m, N)
    return np.nan_to_num(m, 0), N


def get_map(tree, nettree):
    order, layers, ance = tree.bfs(0)
    width = max(layers[-1] - layers[-2], layers[-2] - layers[-3])
    height = len(layers) - 1
    m = np.zeros((height, 0))
    N = np.zeros((height, 0)).astype(str)
    m, N = dfs_(tree, m, N)

    # layers_ = defaultdict(list)
    # k = 0
    # for index, i in enumerate(order):
    #     if index != layers[k]:
    #         layers_[k].append(i)
    #     else:
    #         k += 1
    #         layers_[k].append(i)
    # j = 0
    # for node in layers_[k]:
    #     if node not in order:
    #         continue
    #     father = node
    #     depth=k
    #     while (True):
    #         indices = [i for i, x in enumerate(ance) if x == father]
    #         if len(indices) != 0:
    #             break
    #         father = ance[order.index(father)]
    #         depth-=1
    #     child = [order[i] for i in indices]
    #     child_values = [tree.vs[i]["_nx_name"][1] for i in child]
    #     for c in child:
    #         ance.pop(order.index(c))
    #         order.remove(c)
    #     for cv in child_values:
    #         m[depth][j] = cv
    #         m[depth - 1][j] = np.mean(child_values)
    #         j += 1
    #
    #     x = 3

    return m, N


def otu22d(df, save=False, with_names=False):
    M = []
    for subj in df.iloc:
        nettree = create_tax_tree(subj)
        tree = igraph.Graph.from_networkx(nettree)
        # m = get_map_ieee(tree, nettree)

        m, N = get_map(tree, nettree)
        M.append(m)
        # plt.imshow(m)
        # plt.show()
        if save is not False:
            if with_names is not None:
                save_2d(N, "bact_names", save)
            save_2d(m, subj.name, save)

    if with_names:
        return np.array(M), N
    else:
        return np.array(M)


def otu22d_IEEE(df, save=False):
    M = []
    for subj in df.iloc:
        nettree = create_tax_tree(subj)
        tree = igraph.Graph.from_networkx(nettree)
        m = get_map_ieee(tree, nettree)

        M.append(m)
        # plt.imshow(m)
        # plt.show()
        if save is not False:
            save_2d(m, subj.name, save)
    return np.array(M)


def ppp(p: np.ndarray):
    ret = []
    p = p.astype(float)
    while p.min() < np.inf:
        m = p.argmin()
        ret.append(m)
        p[m] = np.inf

    return ret


def rec(otu, bacteria_names_order, N=None):
    first_row = None
    for i in range(otu.shape[1]):
        if 2 < len(np.unique(otu[0, i, :])):
            first_row = i
            break
    if first_row is None:
        return
    X = otu[:, first_row, :]

    Y = sch.linkage(X.T)
    Z1 = sch.dendrogram(Y, orientation='left')
    idx = Z1['leaves']
    otu[:, :, :] = otu[:, :, idx]
    if N is not None:
        N[:, :] = N[:, idx]

    bacteria_names_order[:] = bacteria_names_order[idx]

    if first_row == (otu.shape[1] - 1):
        return

    unique_index = sorted(np.unique(otu[:, first_row, :][0], return_index=True)[1])

    S = []
    for i in range(len(unique_index) - 1):
        S.append((otu[:, first_row:, unique_index[i]:unique_index[i + 1]],
                  bacteria_names_order[unique_index[i]:unique_index[i + 1]],
                  None if N is None else N[first_row:, unique_index[i]:unique_index[i + 1]]))
    S.append((otu[:, first_row:, unique_index[-1]:], bacteria_names_order[unique_index[-1]:],
              None if N is None else N[first_row:, unique_index[-1]:]))

    for s in S:
        rec(s[0], s[1], s[2])


def dendogram_ordering(otu, df, save=False, N=None):
    names = np.array(list(df.columns))
    rec(otu, names, N)
    df = df[names]
    if save is not False:
        df.to_csv(
            "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7")

    M = []
    if save is not False:
        if N is not None:
            save_2d(N, "bact_names", save)
        for m, index in zip(otu, df.index):
            M.append(m)
            save_2d(m, index, save)
    else:
        for m in otu:
            M.append(m)
    return np.array(M)


def tree_to_newick(g, root=None):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(tree_to_newick(g, root=child))
        else:
            subgs.append(str((child[0][-1], child[1])))
    return "(" + ','.join(subgs) + ")"


def augment(df: pd.DataFrame, layer, length=1000):
    generated = pd.DataFrame(index=df.columns)
    for create_num in range(length):
        subject_a = df.sample(2).iloc[0]
        subject_b = df.sample(2).iloc[1]
        list_of_tax_lists_a = [str(i).split(";") for i in subject_a.index]
        list_of_tax_lists_b = [str(i).split(";") for i in subject_b.index]
        tax_in_layer_a = [row[layer] for row in list_of_tax_lists_a]
        tax_in_layer_b = [row[layer] for row in list_of_tax_lists_b]
        # MAKING  DF
        sample_a = pd.DataFrame(tax_in_layer_a, index=subject_a.index)
        sample_a["values"] = subject_a.values
        sample_b = pd.DataFrame(tax_in_layer_b, index=subject_b.index)
        sample_b["values"] = subject_b.values

        # creating the random samples:
        new_sample = sample_a
        to_change = list(set(random.sample([j for j in new_sample[0]], random.choice(range(len(set(new_sample[0])))))))
        # changing the chosen layers:
        for change in to_change:
            after = list(np.where(new_sample[0] == change, sample_b["values"], new_sample["values"]))
        # df of the new sample
        try:
            new_sample["values"] = after
        except:
            pass
        generated[f'aug{create_num}'] = new_sample["values"]
    return generated.T


if __name__ == '__main__':
    df = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/otus.csv", index_col=0)

    otus2d, names = otu22d(df, False, with_names=True)  # df.iloc[:10]
    dendogram_ordering(otus2d, df, save="2D_otus_dendogram_ordered/Allergy_all_kind_all_times",
                       N=names)
