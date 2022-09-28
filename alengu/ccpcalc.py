""" Conditional clade probability calculation

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from typing import Mapping, Sequence, List, Tuple, Set, FrozenSet, Optional
from collections import defaultdict
from ete3.coretype.tree import TreeNode
from networkx import DiGraph
from alengu.coredata import CCPValue


class CCPCalculations:
    def __init__(self):
        self.gamma_map = defaultdict(
            dict
        )  # type: Mapping[FrozenSet[str], Mapping[frozenset, Sequence[CCPValue]]]
        self.bip_map = defaultdict(lambda: 0)  # type: Mapping[FrozenSet[str], int]
        self.counter = 0  # type: int
        self.root_gkey = None  # type: Optional[TreeNode]
        self.g = DiGraph()  # type: DiGraph
        self.nodes_to_process = set()  # type: Set[TreeNode]
        self.nodes_processed = set()  # type: Set[TreeNode]

    def clear(self):
        self.g = DiGraph()
        self.nodes_to_process = set()
        self.nodes_processed = set()

    def build_multiple_graph(self, gene_trees: Sequence[TreeNode]) -> None:
        for gene_tree in gene_trees:
            self.build_single_graph(gene_tree)
            self.clear()

    def build_single_graph(self, gene_tree: TreeNode) -> None:
        self.counter += 1
        # Build graph and init with leaves
        for i in gene_tree.traverse("levelorder"):
            self.bip_map[i.leaves_key] += 1
            if i.up is not None:
                if i.is_leaf():
                    self.g.add_edge(
                        i.leaves_key,
                        i.up.leaves_key,
                        processed=True,
                        length=i.dist,
                        visited_nodes=i.leaves_key,
                    )
                    self.nodes_to_process.add(i.up.leaves_key)
                    self.nodes_processed.add(i.leaves_key)
                else:
                    self.g.add_edge(
                        i.leaves_key, i.up.leaves_key, processed=False, length=i.dist
                    )
                self.g.add_edge(
                    i.up.leaves_key, i.leaves_key, processed=False, length=i.dist
                )
            else:
                pass  # skip technical root node as it does not reference to any other nod
        self.nodes_to_process.difference(self.nodes_processed)
        self.calcualte_all_edges()
        self.list_builder()

    def calcualte_all_edges(self) -> None:
        nodes_under_processing = set(self.nodes_to_process)
        while len(nodes_under_processing) > 0:
            nodes_to_process = set()
            for node in nodes_under_processing:
                src_data_stamp = [
                    src_node
                    for src_node in self.g.predecessors(node)
                    if self.g[src_node][node]["processed"]
                ]
                dest_data = [
                    dest_node
                    for dest_node in self.g.successors(node)
                    if not self.g[node][dest_node]["processed"]
                ]
                for dest_node in dest_data:
                    src_data = src_data_stamp[:]
                    src_data.sort()
                    if dest_node in src_data:
                        src_data.remove(dest_node)
                    if len(src_data) == 2:
                        proc_edge = self.g[node][dest_node]
                        proc_edge["processed"] = True
                        proc_edge["visited_nodes"] = frozenset.union(
                            self.g[src_data[0]][node]["visited_nodes"],
                            self.g[src_data[1]][node]["visited_nodes"],
                        )
                        proc_edge["src_nodes"] = (
                            self.g[src_data[0]][node]["visited_nodes"],
                            self.g[src_data[1]][node]["visited_nodes"],
                        )
                        proc_edge["length"] = proc_edge["length"]
                        nodes_to_process.add(dest_node)
                    elif len(src_data) > 2:
                        print(f"Warning!: non bifurcating mode {len(src_data)}")

            nodes_under_processing = set(nodes_to_process)

    def add_root_nodes(self, gene_tree: TreeNode) -> None:
        leaf_names = sorted([leaf.leaves_key for leaf in gene_tree.get_leaves()])
        self.root_gkey = frozenset.union(*leaf_names)
        nodes_l = list(self.gamma_map.keys())
        for node in nodes_l:
            leaf_src = (node, self.root_gkey - node)
            leaf_src_rev = (self.root_gkey - node, node)
            if leaf_src_rev not in self.gamma_map[self.root_gkey]:
                self.gamma_map[self.root_gkey][leaf_src] = CCPValue(
                    self.root_gkey, leaf_src, self.bip_map[node]
                )
            else:
                self.gamma_map[self.root_gkey][leaf_src_rev].score += self.bip_map[node]

    def list_builder(self):
        edge_list = list(self.g.edges(data=True))
        for src, dest, e in edge_list:
            gamma_src_key = e["src_nodes"] if "src_nodes" in e else None
            if gamma_src_key is not None:
                gamma_src_key_r = tuple(reversed(gamma_src_key))
            else:
                gamma_src_key_r = None
            if gamma_src_key in self.gamma_map[e["visited_nodes"]]:
                self.gamma_map[e["visited_nodes"]][gamma_src_key] = CCPValue(
                    e["visited_nodes"],
                    e["src_nodes"] if "src_nodes" in e else None,
                    1 + self.gamma_map[e["visited_nodes"]][gamma_src_key].score,
                )
            elif gamma_src_key_r in self.gamma_map[e["visited_nodes"]]:
                self.gamma_map[e["visited_nodes"]][gamma_src_key_r] = CCPValue(
                    e["visited_nodes"],
                    e["src_nodes"] if "src_nodes" in e else None,
                    1 + self.gamma_map[e["visited_nodes"]][gamma_src_key_r].score,
                )
            else:
                self.gamma_map[e["visited_nodes"]][gamma_src_key] = CCPValue(
                    e["visited_nodes"], e["src_nodes"] if "src_nodes" in e else None, 1
                )

    def calculate_norm(self):
        for g, gamma_list in self.gamma_map.items():
            gamma_sum = sum([gamma.score for gamma in gamma_list.values()])
            # if g != self.root:
            for gamma in gamma_list.values():
                if gamma.score != 0:
                    if g != self.root_gkey:
                        gamma.norm_score = gamma.score / gamma_sum
                    else:
                        gamma.norm_score = gamma.score / self.counter
                else:
                    gamma.score = self.counter
                    gamma.norm_score = 1.0

    def convert_map_to_list(
        self, gamma_map: Mapping[frozenset, Mapping[frozenset, CCPValue]]
    ) -> Sequence[Tuple[frozenset, Mapping[frozenset, CCPValue]]]:
        return sorted(gamma_map.items(), key=lambda n: (len(n[0]), n[0]))

    def get_gamma(self) -> Sequence[TreeNode]:
        gamma_list = self.convert_map_to_list(self.gamma_map)
        return gamma_list

    def get_gamma_batched(self) -> Sequence[Sequence[Tuple[TreeNode]]]:
        gamma_list_and_size = [(len(n[0]), n) for n in self.gamma_map.items()]
        max_s = max(([size for size, g in gamma_list_and_size]))
        min_s = min([size for size, g in gamma_list_and_size])
        batch = []  # type: List[List[TreeNode]]
        for i in range(max_s):
            batch.append([])
        for size, g in gamma_list_and_size:
            batch[size - 1].append(g)
        return batch

    def make_gid_maps(self):
        i_to_n = dict()
        n_to_i = dict()
        nodes = sorted(self.gamma_map, key=lambda i: (len(i), i))
        count = 0
        for n in nodes:
            i_to_n[count] = n
            n_to_i[n] = count
            count += 1
        return i_to_n, n_to_i, count

    def add_calculations_to_indexes(self, loader, indexes):
        self.build_multiple_graph(loader.gene_trees)
        self.add_root_nodes(loader.gene_trees[0])
        self.calculate_norm()
        indexes.gkey_neighbours_list = self.get_gamma()
        indexes.gkey_neighbours_list_b = self.get_gamma_batched()
        indexes.root_gkey = self.root_gkey
