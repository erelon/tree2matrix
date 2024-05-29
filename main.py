import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import scipy.cluster.hierarchy as sch

from utils import generate_random_tree, get_root, rand_tree_values


class MatricesDendrogram:
    def dendogramize(self, map_from_tree, N=None):
        first_row = None
        for i in range(map_from_tree.shape[1]):
            if 2 < len(np.unique(map_from_tree[0, i, :])):
                first_row = i
                break
        if first_row is None:
            return
        X = map_from_tree[:, first_row, :]

        Y = sch.linkage(X.T)

        sys.setrecursionlimit(13000)
        MB = 2 ** 20
        threading.stack_size(MB * 64)
        tpe = ThreadPoolExecutor(1)

        f = tpe.submit(sch.dendrogram, Y, orientation='left', no_plot=True)
        Z1 = f.result()

        idx = Z1['leaves']
        map_from_tree[:, :, :] = map_from_tree[:, :, idx]
        if N is not None:
            N[:, :] = N[:, idx]

        if first_row == (map_from_tree.shape[1] - 1):
            return

        unique_index = sorted(np.unique(map_from_tree[:, first_row, :][0], return_index=True)[1])

        S = []
        for i in range(len(unique_index) - 1):
            S.append((map_from_tree[:, first_row:, unique_index[i]:unique_index[i + 1]],
                      # bacteria_names_order[unique_index[i]:unique_index[i + 1]],
                      None if N is None else N[first_row:, unique_index[i]:unique_index[i + 1]]))
        S.append((map_from_tree[:, first_row:, unique_index[-1]:],  # bacteria_names_order[unique_index[-1]:],
                  None if N is None else N[first_row:, unique_index[-1]:]))

        for s in S:
            self.dendogramize(s[0], s[1])

    def transform(self, matrices: list, matrices_names):
        matrices = np.array(matrices)
        # Dendrogram
        self.dendogramize(matrices,  # names,
                          matrices_names)
        return matrices, matrices_names


class Tree2Matrix:
    #
    def dfs_rec(self, node, depth, m, N, counter):
        c = counter[0]
        self.added[node] = True
        successors = self.tree.successors(node)
        total_value = 0
        num_of_sons = 0
        num_of_descendants = 0
        for successor in successors:
            if not self.added[successor]:
                val, m, descendants, N, dec_name = self.dfs_rec(successor, depth + 1, m, N, counter)
                total_value += val
                num_of_sons += 1
                num_of_descendants += descendants

        if num_of_sons == 0:
            value = self.tree.nodes[node][self.value_key]
            m[depth, c] = value

            if self.name_key is None:
                name = node
            else:
                name = self.tree.nodes[node][self.name_key]
            N[depth, c] = name
            counter[0] += 1

            return value, m, 1, N, name

        if self.name_key is None:
            name = node
        else:
            name = self.tree.nodes[node][self.name_key]

        if self.aggregation == "avg":
            node_value = total_value / num_of_sons
        elif self.aggregation == "sum":
            node_value = total_value

        for j in range(num_of_descendants):
            m[depth][c + j] = node_value
            N[depth][c + j] = name

        return node_value, m, num_of_descendants, N, name

    def dfs_(self, m, N):
        self.added = {node: False for node in self.tree.nodes}
        counter = [0]
        _, m, _, N, _ = self.dfs_rec(get_root(self.tree), 0, m, N, counter)
        return np.nan_to_num(m, 0), N

    def transform(self, tree: nx.DiGraph, value_key: str, name_key: str = None, aggregation="avg"):
        self.tree = tree
        self.value_key = value_key
        self.name_key = name_key
        if aggregation not in ["avg", "sum"]:
            raise ValueError("Only 'avg' and 'sum' are valid as aggregation methods.")
        self.aggregation = aggregation

        # Tree 2 map
        layers = list(nx.bfs_layers(self.tree, get_root(self.tree)))
        height = len(layers)
        leafs_num = len([i for i in self.tree if len(self.tree.succ[i]) == 0])

        map_from_tree = np.zeros((height, leafs_num)) / 0
        if name_key is None:
            str_size = max([len(str(i)) for i in self.tree]) + 16
        else:
            str_size = max([len(str(self.tree.nodes[i][name_key])) for i in self.tree]) + 16

        Names_in_map = np.zeros((height, leafs_num)).astype(f"U{str_size}")
        map_from_tree, Names_in_map = self.dfs_(map_from_tree, Names_in_map)
        return map_from_tree, Names_in_map


if __name__ == '__main__':
    while True:
        tree = generate_random_tree(10)
        if len(tree.nodes) >= 50:
            break

    M = []
    t2m = Tree2Matrix()
    for i in range(5):
        t = rand_tree_values(copy.deepcopy(tree))
        m, N = t2m.transform(t, "val", "name")
        M.append(m)

    md = MatricesDendrogram()
    matrices_result, new_order_names = md.transform(M, N)
