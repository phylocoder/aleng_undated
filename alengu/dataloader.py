"""This module is responsible for loading data

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from typing import Optional, Sequence, Mapping
from ete3 import Tree
from ete3.parser.newick import NewickError
from ete3.coretype.tree import TreeNode
from alengu.aleloader import ALELoader


class DataLoader:
    """This class loads data files from the disk and pre-process them"""

    def __init__(self):
        self.gene_trees = []  # type: Sequence[TreeNode]
        self.species_tree = None  # type: Optional[TreeNode]
        self.gene_species_map = {}
        self.need_ccp = False
        self.gamma = None
        self.gamma_b = None
        self.root = None
        self.gid_to_node = None
        self.gid_size = 0
        self.node_to_gid = None
        self.s_node_name = None
        
        

    def init_from_files(self, s_filename: str, g_filename: str) -> None:
        """Load files from disk"""
        with open(s_filename, "rt", encoding="utf8") as sf:
            s = sf.readline()
        self.add_species_tree(s)
        with open(g_filename, "rt", encoding="utf8") as gf:
            g = []
            for line in gf:
                g.append(line)
        for line in g:
            self.add_gene_tree(line)
        self.need_ccp = True
        # self.construct_gene_s_map()

    def construct_gene_s_map(
        self, gid_to_names: Mapping[int, str], separator: str = "_"
    ) -> None:
        for v in gid_to_names.values():
            if v != "":
                name_base = v.split(separator)[0]
                self.gene_species_map[frozenset({v})] = frozenset({name_base})

    def init_from_files_ale(self, s_filename: str, g_filename: str) -> None:
        """Load files from disk"""
        with open(s_filename, "rt", encoding="utf8") as sf:
            s = sf.readline()
        self.add_species_tree(s)
        a = ALELoader()
        a.get_gamma_from_file(g_filename)
        self.gamma = a.gamma_map
        self.gamma_b = a.get_gamma_batched()
        self.gid_to_node, self.node_to_gid, self.gid_size = a.make_gid_maps()
        self.construct_gene_s_map(a.gid_to_names)
        self.root = a.root

    def init_from_str(self, s: str, g: Sequence[str]) -> None:
        self.add_species_tree(s)
        for line in g:
            self.add_gene_tree(line)

    def add_gene_trees_from_file(self, gene_file_name: str) -> None:
        with open(gene_file_name, "rt", encoding="utf8") as ifile:
            for row in ifile:
                gene_tree = Tree(row, format=1)
                self.unroot_keep(gene_tree)
                self.default_name_fixer(gene_tree)
                self.gene_trees.append(gene_tree)

    def check_tree(self, gene_tree):
        leaves = set()
        for node in gene_tree.get_leaves():
            if node.name not in leaves:
                leaves.add(node.name)
            else:
                print("WARNING", node.name)

    def add_gene_tree(self, tree_string: str) -> None:
        try:
            gene_tree = Tree(tree_string, format=1)  # , is_file=False
        except NewickError:
            raise ValueError(
                "File contains the gene tree is not a proper newick format. Do you want to load an ALE file?"
            )
        self.check_tree(gene_tree)
        self.unroot_keep(gene_tree)
        self.spec_name_cutter(gene_tree)
        self.default_name_fixer(gene_tree)
        self.gene_trees.append(gene_tree)

    def add_species_tree(self, tree_string: str) -> None:
        """Initialize species_tree var"""
        species_tree = Tree(tree_string, format=1)
        self.normalize_tree(species_tree)
        self.default_name_fixer(species_tree)
        self.species_tree = species_tree
        self.s_node_name = self.create_node_key_name(species_tree)

    @staticmethod
    def unroot_keep(tree):
        if len(tree.children) == 2:
            if not tree.children[0].is_leaf():
                tree.children[1].dist += tree.children[0].dist
                tree.children[0].delete()
            elif not tree.children[1].is_leaf():
                tree.children[0].dist += tree.children[1].dist
                tree.children[1].delete()
            else:
                raise ValueError("Cannot unroot a tree with only two leaves")

    @staticmethod
    def default_name_fixer(tree):
        """generate default names"""
        for i in tree.traverse("postorder"):
            if i.name != "":
                i.leaves_key = frozenset((i.name,))
            if i.name == "":
                i.leaves_key = frozenset.union(*(v.leaves_key for v in i.children))
                i.name = "-".join(sorted(i.leaves_key))

    @staticmethod
    def create_node_key_name(tree):
        map_n_k = {}
        for i in tree.traverse():
            map_n_k[i.leaves_key] = i.name
        return map_n_k

    def spec_name_cutter(self, tree: TreeNode, separator="_") -> None:
        """generate name_base attribute"""
        for i in tree.traverse("postorder"):
            if i.name != "":
                i.name_base = i.name.split(separator)[0]
                self.gene_species_map[frozenset({i.name})] = frozenset({i.name_base})

    def get_avg_height(self, tree):
        leaves = tree.get_leaves()
        dist = [tree.get_distance(n) for n in leaves]
        return sum(dist) / len(dist)

    def normalize_tree(self, tree):
        height = self.get_avg_height(tree)
        for i in tree.traverse():
            if i == tree:
                i.dist = 1.0
            else:
                i.dist /= height
