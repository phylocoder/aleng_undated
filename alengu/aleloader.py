"""This module is responsible for ALE file parsing

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from collections import defaultdict
from typing import Sequence, Mapping, Optional
from alengu.coredata import CCPValue


class ALELoader:
    """Loads an ALE file

    Attributes:
        gamma_map: a 2D map of Gamma objects: first key identifies the node and the second one its children
        gid_to_counts: A map from gid to its occurence count
        gid_to_end: ...
        dip_counts: ...
        gid_to_names: ...
        set_ids: ...
        observ:

    """

    CONSTRUCTOR_STR = "#constructor_string"  # type:str
    OBSERV_STR = "#observations"  # type:str
    BIP_STR = "#Bip_counts"  # type:str
    BIP_BLS_STR = "#Bip_bls"  # type:str
    DIP_STR = "#Dip_counts"  # type:str
    LAST_LEAFSET_ID_STR = "#last_leafset_id"  # type:str
    LEAF_ID_STR = "#leaf-id"  # type:str
    SET_ID_STR = "#set-id"  # type:str
    END_STR = "#END"  # type:str

    def __init__(self) -> None:
        self.gamma_map = defaultdict(dict)  # type: Mapping[Mapping]
        self.gid_to_counts = None  # type: Optional[Mapping[int, int]]
        self.gid_to_end = None  # type: Optional[Mapping]
        self.dip_counts = None
        self.gid_to_names = None
        self.set_ids = None
        self.observ = None  # type: Optional[int]
        self.last_leafset_id = None
        self.gene_species_mapper = {}
        self.gid_to_key = None
        self.root = None

    def is_data_loaded(self) -> bool:
        """check whether ALE data successfully loaded or not"""
        return (
            self.gid_to_counts is not None
            and self.gid_to_end is not None
            and self.dip_counts is not None
            and self.gid_to_names is not None
            and self.set_ids is not None
            and self.observ is not None
        )

    def load_ale(self, file_name: str) -> None:
        """the ale file parser"""

        bip_counts = {}  # type: Mapping[int, int]
        bip_bls = {}  # type: Mapping[int, float]
        dip_counts = {}
        leaf_ids = {}
        set_ids = {}  # type: Mapping[int, Sequence[int]]
        observ = None  # type: Optional[int]
        with open(file_name) as ale:
            cmd = ""
            for row in ale:
                if row[0] == "#":
                    cmd = row[:-1]
                elif cmd == self.CONSTRUCTOR_STR:
                    pass
                elif cmd == self.OBSERV_STR:
                    observ = int(row)
                elif cmd == self.BIP_STR:
                    num_list = [int(num) for num in row.split("\t")]
                    if len(num_list) != 2:
                        raise ValueError("Bip_counts lines must contain two integers!")
                    node_id, node_count = num_list
                    bip_counts[node_id] = node_count
                elif cmd == self.BIP_BLS_STR:
                    node_id, node_value = row.split("\t")
                    bip_bls[int(node_id)] = float(node_value)
                elif cmd == self.DIP_STR:
                    parent, left, right, count = [int(num) for num in row.split("\t")]
                    dip_counts[(parent, left, right)] = count
                elif cmd == self.LAST_LEAFSET_ID_STR:
                    last_leafset_id = int(row)
                elif cmd == self.LEAF_ID_STR:
                    parent, count = row.split("\t")
                    leaf_ids[int(count)] = parent
                elif cmd == self.SET_ID_STR:
                    parent, separator, *count = row.split("\t")
                    set_ids[int(parent)] = [int(num) for num in count]
                elif cmd == self.END_STR:
                    pass
                else:
                    raise ValueError("Invalid ALE format")

        for parent, name in leaf_ids.items():
            bip_counts[parent] = observ
        self.last_leafset_id = last_leafset_id
        self.gid_to_counts = bip_counts
        self.gid_to_end = bip_bls
        self.dip_counts = dip_counts
        self.gid_to_names = leaf_ids
        self.set_ids = set_ids
        self.observ = observ

    @staticmethod
    def create_gid_to_key(set_id_to_leaf_ids, leaf_id_to_leaf_name):
        """TODO: document"""
        id_map = {}
        for set_id, leaf_ids in set_id_to_leaf_ids.items():
            id_map[set_id] = frozenset((leaf_id_to_leaf_name[i] for i in leaf_ids))
        return id_map

    def create_gamma_map(
        self, gid_to_key, dip_counts, bip_counts, observ, gid_to_names
    ):
        """TODO: document"""
        key_to_gid = {}
        # prepare reverse map
        for gamma_id, gamma_key in gid_to_key.items():
            key_to_gid[gamma_key] = gamma_id
        # add bi partitions
        for (gid, gid_c1, gid_c2), count in dip_counts.items():
            source_ids = (gid_to_key[gid_c1], gid_to_key[gid_c2])
            self.gamma_map[gid_to_key[gid]][source_ids] = CCPValue(
                gid_to_key[gid], source_ids, count, count / bip_counts[gid]
            )
        # add leaves
        for key in gid_to_names.values():
            leaf = frozenset({key})
            self.gamma_map[leaf][None] = CCPValue(leaf, None, observ, 1.0)
        # add root
        self.root = frozenset(gid_to_names.values())
        nodes_l = list(self.gamma_map.keys())
        for node in nodes_l:
            leaf_src = (node, self.root - node)
            leaf_src_rev = (self.root - node, node)
            if leaf_src_rev not in self.gamma_map[self.root]:
                new_gamma_instance = CCPValue(
                    self.root,
                    leaf_src,
                    bip_counts[key_to_gid[node]],
                    bip_counts[key_to_gid[node]] / observ,
                )
                self.gamma_map[self.root][leaf_src] = new_gamma_instance

    def get_gamma_from_file(self, filename):
        """load the file, then calculates maps"""
        self.load_ale(filename)
        self.gid_to_key = self.create_gid_to_key(self.set_ids, self.gid_to_names)
        self.create_gamma_map(
            self.gid_to_key,
            self.dip_counts,
            self.gid_to_counts,
            self.observ,
            self.gid_to_names,
        )

    def get_gamma_batched(self) -> Sequence[Sequence[CCPValue]]:
        # gamma_list=sorted(self.gamma_map.items(), key=lambda n:(len(n[0]), n[0]))
        gamma_list_and_size = [(len(n[0]), n) for n in self.gamma_map.items()]
        max_s = max(([size for size, gamma in gamma_list_and_size]))
        batch = [[]] * max_s  # type: List[List[TreeNode]]
        for size, gamma in gamma_list_and_size:
            batch[size - 1].append(gamma)
        return batch

    def make_gid_maps(self):
        i_to_n = dict()
        n_to_i = dict()
        nodes = sorted(self.gamma_map, key=lambda i: (len(i), i))
        count = 0
        for node in nodes:
            i_to_n[count] = node
            n_to_i[node] = count
            count += 1
        return i_to_n, n_to_i, count

    def construct_gene_s_map(self, leaf_id):
        """generate gene name to species name map"""
        for name in leaf_id.values():
            name_base = name.split("_")[0]
            self.gene_species_mapper[frozenset({name})] = frozenset({name_base})
