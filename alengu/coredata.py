""" Core data structures used by ALE NG 

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from collections import defaultdict
from typing import Mapping, Sequence, Optional, Tuple, NewType
from ete3 import TreeNode

SpeciesKey = NewType("SpeciesKey", frozenset[str])
GeneKey = NewType("GeneKey", frozenset[str])
SpeciesId = NewType("SpeciesId", int)
GeneId = NewType("GeneId", int)
RankId = NewType("RankId", int)
SliceId = NewType("SliceId", int)


class CCPValue:
    """Conditional clade probabability data"""

    def __init__(self, key, children, score, norm_score=None):
        self.key = key
        self.children = children
        self.score = score
        if norm_score is None:
            self.norm_score = score
        else:
            self.norm_score = norm_score

    def __repr__(self):
        return str(self)

    def __str__(self):
        k = str(list(self.key))
        if self.children is not None and len(self.children) == 2:
            ch1 = str(list(self.children[0]))
            ch2 = str(list(self.children[1]))
        else:
            ch1 = ch2 = "---"
        return f"<Gamma: {k} => {ch1}-{ch2}, ns: {self.norm_score}, s: {self.score}>"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.key == other.key
                and (
                    self.children == other.children
                    or self.children == (other.children[1], other.children[0])
                )
                and self.norm_score == other.norm_score
            )
        return False


class ExtinctionData:
    """Class to contain extintion related calculation"""

    def __init__(self, e, h_hat):
        self.e = e
        self.h_hat = h_hat

    def __iter__(self):
        yield self.e
        yield self.h_hat


class ExtinctionDataU:
    def __init__(self, e, e_hat, e_hat_c):
        self.e = e
        self.e_hat = e_hat
        self.e_hat_c = e_hat_c

    def __iter__(self):
        yield self.e
        yield self.e_hat
        yield self.e_hat_c


class ProbabilityData:
    def __init__(self, p, p_hat):
        self.p = p
        self.p_hat = p_hat

    def __iter__(self):
        yield self.p
        yield self.p_hat


class ProbabilityDataU:
    def __init__(self, p, p_hat, p_hat_c):
        self.p = p
        self.p_hat = p_hat
        self.p_hat_c = p_hat_c

    def __iter__(self):
        yield self.p
        yield self.p_hat
        yield self.p_hat_c


class Indexes:
    def __init__(
        self,
        root_gkey: GeneKey = None,
        rank_to_start_pos: Mapping[int, float] = None,
        rank_to_time_slice_steps: Mapping[int, float] = None,
        slice_to_node=None,
        gkey_to_skey=None,
        sliceid_to_len=None,
        model_params=None,
    ):
        self.root_gkey = root_gkey  # type: frozenset
        self.rank_to_start_pos = rank_to_start_pos  # type: Sequence[float]
        self.rank_to_rankdata = None
        self.rank_to_time_slice_steps = (
            rank_to_time_slice_steps
        )  # type: Sequence[TreeCalculation.TimeSliceSteps]
        self.sliceids_to_node = (
            slice_to_node
        )  # type: Mapping[Tuple[int, int],Sequence[TreeNode]]
        self.gkey_to_skey = gkey_to_skey  # type: Mapping[frozenset, frozenset]
        self.skey_to_node = None  # type: Mapping[frozenset, TreeNode]
        self.sliceid_to_len = sliceid_to_len  # type: Mapping[Tuple[int, int],int]
        self.model_params = model_params  # type: Optional[ModelParams]
        self.rank_to_active_node = (
            None
        )  # type: Sequence[Sequence[Tuple[TreeNode,boolean]]]
        self.loop_list_mod = (
            None
        )  # type: Tuple[float, int, int, Sequence[Tuple[TreeNode,boolean]]]
        self.species_nodes = None
        self.skey_to_ancestors = None
        self.gkey_neighbours_list = (
            None
        )  # type: Sequence[Tuple[frozenset, Mapping[frozenset, CCPValue]]]
        self.gkey_neighbours_list_b = (
            None
        )  # type: Sequence[Tuple[frozenset, Mapping[frozenset, CCPValue]]]
        self.gkey_to_name = None
        self.s_node_name = None

    def create_gkey_to_name(self, gamma):
        counter = 1
        gkey_to_name = {}
        for k, ks in gamma:
            gkey_to_name[k] = tuple(k)[0] if len(k) == 1 else "N" + str(counter)
            counter += 1
        return gkey_to_name


class ModelParams:
    """Contains the parameters of the computed model"""

    def __init__(
        self,
        delta: float = 0.01,
        tau: float = 1.0,  # 0.01,
        p_lambda: float = 0.01,
        sigma_hat: float = 1.0,
    ) -> None:
        self.duplication_rate = delta  # type: float
        self.transfer_rate = tau  # type: float
        self.loss_rate = p_lambda  # type: float
        self.speciation_rate = sigma_hat  # type: float

    @property
    def delta(self) -> float:
        return self.duplication_rate

    @property
    def p_lambda(self) -> float:
        return self.loss_rate

    @property
    def tau(self) -> float:
        return self.transfer_rate

    @property
    def sigma_hat(self) -> float:
        return self.speciation_rate

    def __str__(self) -> str:
        return f"delta={self.duplication_rate}, tau={self.transfer_rate}, lambda={self.loss_rate}"


def convert_node_to_name(node):
    return "-".join(sorted(list(node)))


class MatrixIndexes:
    """Indexes for the matrix based calculation"""

    def __init__(self):
        self.rank_size = 0
        self.ts_size = 0
        self.ts_len_vect = 0
        self.sid_to_node = None
        self.node_to_sid = None
        self.key_to_sid = None
        self.sid_size = None
        self.act_node_list = None
        self.loop_list_mod_vect = None
        self.loop_list_vect = None
        self.nodes_in_slices = None
        self.len_of_slices = None
        self.delta_t_vect = None
        self.gid_to_node = None
        self.node_to_gid = None
        self.gid_size = None
        self.gid_to_sid = None


class SamplerStat:
    """Sampling statistics"""

    def __init__(self):
        self.branch_counts_copies = defaultdict(int)
        self.branch_counts_saw = defaultdict(int)
        self.branch_counts_presence = defaultdict(int)
        self.branch_counts_singleton = defaultdict(int)
        self.branch_roots = defaultdict(int)
        self.branch_counts_count = defaultdict(int)
        self.branch_Ds = defaultdict(int)
        self.branch_Ts = defaultdict(int)
        self.branch_Tfroms = defaultdict(int)
        self.branch_Ls = defaultdict(int)
        self.T_to_from = defaultdict(int)
        self.D = 0
        self.T = 0
        self.L = 0
        self.S = 0

    def __str__(self):
        return "D: {}, T: {}, L: {}, S: {}".format(self.D, self.T, self.L, self.S)
