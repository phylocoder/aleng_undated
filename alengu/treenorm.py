""" Can normailze the tree for a unit height

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from ete3 import Tree


def load_tree(fname):
    with open(fname, "rt", encoding="utf8") as sf:
        s = sf.readline()
    return Tree(s, format=1)


def print_length_stat(tree):
    # print(tree)
    leaves = tree.get_leaves()
    dist = [tree.get_distance(n) for n in leaves]
    print("Lengths: ", dist)
    print("Avg: ", sum(dist) / len(dist))


def get_avg_height(tree):
    leaves = tree.get_leaves()
    dist = [tree.get_distance(n) for n in leaves]
    return sum(dist) / len(dist)


def normalize_tree(tree):
    height = get_avg_height(tree)
    for i in tree.traverse():
        if i == tree:
            i.dist = 1.0
        else:
            i.dist /= height


if __name__ == "__main__":
    su_filename = "S.tree"
    s_filename = "Snorm.tree"
    s_tree = load_tree(s_filename)
    su_tree = load_tree(su_filename)
    print_length_stat(su_tree)
    print_length_stat(s_tree)
    normalize_tree(su_tree)
    print_length_stat(su_tree)
    su_tree.write(format=1, outfile="S_fix.tree")
