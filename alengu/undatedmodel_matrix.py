""" Matrix based calulation of the undated model

The functions moved in front (outside of the class definitions) to enable NUMBA compilation

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
import math
from typing import Sequence
from time import perf_counter
from numba import njit, f8, u1, u4, b1
import numpy as np
from alengu.coredata import ModelParams, Indexes, MatrixIndexes

EPSILON = np.finfo(np.double).eps
ITER_NUM = 4


# @timing
@njit(locals={"P_res": f8[:, ::1], "i": u1}, nogil=True, cache=True, fastmath=True)
def compute_core(
    E,
    E_hat,
    E_hat_c,
    P,
    P_hat,
    P_hat_c,
    ancestors,
    u_gid,
    gid_sid_map,
    norms,
    p_d,
    p_s,
    p_t,
    left,
    right,
    vw_gids,
    w,
    a_has_children,
    a_left,
    a_right,
):
    """The main entry point of the probability computation"""
    for i in range(ITER_NUM):
        P_res = pcalc_base(
            E,
            E_hat,
            E_hat_c,
            P,
            P_hat,
            P_hat_c,
            ancestors,
            u_gid,
            gid_sid_map,
            p_d,
            p_s,
            p_t,
            left,
            right,
            w,
            a_has_children,
            a_left,
            a_right,
        )
        for j in range(len(norms)):
            v_gid = vw_gids[j, 0]
            w_gid = vw_gids[j, 1]
            norm = norms[j]
            p = pcalc_iter(
                P,
                P_hat,
                P_hat_c,
                norm,
                p_d,
                p_s,
                left,
                right,
                v_gid,
                w,
                w_gid,
                a_has_children,
                a_left,
                a_right,
            )
            P_res += p
            pass
        # store_fcall(E, E_hat, E_hat_c,
        #            P, P_hat, P_hat_c,
        #            ancestors, gamma_id, gid_sid_map, norms,
        #            p_d, p_s, p_t, left, right, vw_gids, w, a_has_children, a_left, a_right)

        P[u_gid, :] = P_res


@njit(
    f8[::1](f8[::1], b1[::1], u4[::1]),
    locals={"res": f8[::1]},
    nogil=True,
    cache=True,
    fastmath=True,
)
def select_child(src, a_has_children, a_side):
    """Child selection vector construction"""
    res = np.zeros(a_side.shape, dtype=np.float64)
    for i in range(len(a_has_children)):
        if a_has_children[i]:
            res[i] = src[a_side[i]]
    return res


@njit(
    f8[:, ::1](
        f8[:, ::1],  # E
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],  # P
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],  # ancestors
        u4,
        u1[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        b1[::1],
        u4[::1],
        u4[::1],
    ),
    locals={"P_res": f8[:, ::1]},
    nogil=True,
    cache=True,
    fastmath=True,
)
def pcalc_base(
    E,
    E_hat,
    E_hat_c,
    P,
    P_hat,
    P_hat_c,
    ancestors,
    u_gid,
    gid_sid_map,
    p_d,
    p_s,
    p_t,
    left,
    right,
    w,
    a_has_children,
    a_left,
    a_right,
):
    """Calculation of the probability of speciation + loss, transfer + loss and duplication + loss events"""
    P_hat[u_gid] = (P[u_gid, :] * p_t).sum()
    P_hat_c[u_gid, :] = (P[u_gid, :] @ ancestors) * p_t
    P_res = gid_sid_map[u_gid, :] * p_s
    P_res += (
        P[u_gid, :] * (E_hat - E_hat_c) + E * (P_hat[u_gid] - P_hat_c[u_gid, :])
    ) / w
    # DL event
    P_res += p_d * 2 * P[u_gid, :] * E
    # SL events
    P_res += p_s * (
        (select_child(E[0, :], a_has_children, a_left))
        * (select_child(P[u_gid, :], a_has_children, a_right))
        + (select_child(P[u_gid, :], a_has_children, a_left))
        * (select_child(E[0, :], a_has_children, a_right))
    )
    return P_res


@njit(
    f8[:, ::1](
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        f8,
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        f8[:, ::1],
        u4,
        f8[:, ::1],
        u4,
        b1[::1],
        u4[::1],
        u4[::1],
    ),
    locals={"P_res": f8[:, ::1]},
    nogil=True,
    cache=True,
    fastmath=True,
)
def pcalc_iter(
    P,
    P_hat,
    P_hat_c,
    norm,
    p_d,
    p_s,
    left,
    right,
    v_gid,
    w,
    w_gid,
    a_has_children,
    a_left,
    a_right,
):
    """Calculation of the probability of speciations, duplication and transfer events"""
    # S events
    P_res = (
        norm
        * p_s
        * (
            (select_child(P[v_gid, :], a_has_children, a_right))
            * (select_child(P[w_gid, :], a_has_children, a_left))
            + (select_child(P[w_gid, :], a_has_children, a_right))
            * (select_child(P[v_gid, :], a_has_children, a_left))
        )
    )
    # D and T events
    P_res += norm * (
        p_d * (P[v_gid, :] * P[w_gid, :])
        + (P_hat[w_gid] - P_hat_c[w_gid, :]) / w * P[v_gid, :]
        + (P_hat[v_gid] - P_hat_c[v_gid, :]) / w * P[w_gid, :]
    )
    return P_res


def store_fcall(
    E,
    E_hat,
    E_hat_c,
    P,
    P_hat,
    P_hat_c,
    ancestors,
    u_gid,
    gid_sid_map,
    norms,
    p_d,
    p_s,
    p_t,
    left,
    right,
    vw_gids,
    w,
    a_has_children,
    a_left,
    a_right,
):
    """Debug function to dump the state of a computational step"""
    ts = perf_counter()
    with open(f"datadump_{u_gid}_{ts}.npz", "wb") as dumpf:
        np.savez(
            dumpf,
            E=E,
            E_hat=E_hat,
            E_hat_c=E_hat_c,
            P=P,
            P_hat=P_hat,
            P_hat_c=P_hat_c,
            ancestors=ancestors,
            u_gid=u_gid,
            gid_sid_map=gid_sid_map,
            norms=norms,
            p_d=p_d,
            p_s=p_s,
            p_t=p_t,
            left=left,
            right=right,
            vw_gids=vw_gids,
            w=w,
            a_has_children=a_has_children,
            a_left=a_left,
            a_right=a_right,
        )


class UndatedModelMatrix:
    """Undated Model coputation functions using matrix representation"""

    def __init__(self, conf: Indexes, mconf: MatrixIndexes, param: ModelParams):
        self.e_mode = False  # type: bool
        self.conf = conf  # type: Indexes
        self.listofnodes = self.conf.species_nodes
        self.param = param  # type: ModelParams
        self.mapping = conf.gkey_to_skey
        self.sid_size = mconf.sid_size  # type: int
        self.gid_size = mconf.gid_size
        self.ts_len_vect = mconf.ts_len_vect  # type: Sequence[int]
        self.nodes_in_slices = mconf.nodes_in_slices  # type: Sequence[bool]
        self.len_of_slices = mconf.len_of_slices  # type: Sequence[int]
        self.delta_t_vect = np.array(
            mconf.delta_t_vect, dtype=np.double
        )  # type: Sequence[float]
        self.gid_to_sid_map = mconf.gid_to_sid_map  # type: Sequence[int]
        self.node_to_gid = mconf.node_to_gid
        self.node_to_sid = mconf.node_to_sid
        self.s_gene = conf.root_gkey
        self.left, self.right = self.create_left_right(self.listofnodes)
        self.a_has_children, self.a_left, self.a_right = self.create_left_right_arrays(
            self.listofnodes
        )
        self.ancestors = self.create_ancestor_map(self.listofnodes)
        self.gid_sid_map = self.create_gid_sid_mapping()
        self.w = self.calculate_node_weight(self.ancestors, self.listofnodes)
        if param is not None:
            row_vector = np.ones((1, self.sid_size), dtype=np.double)  # TYPEDOUBLE
            self.p_s = row_vector * (
                1 / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
            self.p_d = row_vector * (
                param.duplication_rate
                / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
            self.p_t = row_vector * (
                param.transfer_rate
                / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
            self.p_l = row_vector * (
                param.loss_rate
                / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
        self.last_ext_data = None
        self.last_prob_data = None
        self.opt_counter = 0

    def create_ancestor_map(self, listofnodes):
        """Construction matrix of ancestor ids"""
        ancestors = np.zeros((self.sid_size, self.sid_size), dtype=np.double)

        for node in listofnodes:
            node_sid = self.node_to_sid[node]
            ancestors[node_sid, node_sid] = 1
            next_node = node.up
            while next_node is not None:
                ancestors[self.node_to_sid[next_node], node_sid] = 1
                next_node = next_node.up
        return ancestors

    def create_left_right(self, listofnodes):
        """Creating matrixes to mask children in computation"""
        left = np.zeros((self.sid_size, self.sid_size), dtype=np.double)
        right = np.zeros((self.sid_size, self.sid_size), dtype=np.double)

        for node in listofnodes:
            if len(node.children) == 2:
                node_sid = self.node_to_sid[node]
                left[self.node_to_sid[node.children[0]], node_sid] = 1
                right[self.node_to_sid[node.children[1]], node_sid] = 1

        return left, right

    def create_left_right_arrays(self, listofnodes):
        """Creating vectors of children ids and a vector of children presence"""
        left = np.zeros((self.sid_size,), dtype=np.uint32)
        right = np.zeros((self.sid_size,), dtype=np.uint32)
        has_children = np.zeros((self.sid_size,), dtype=np.bool)
        for node in listofnodes:
            if len(node.children) == 2:
                node_sid = self.node_to_sid[node]
                has_children[node_sid] = True
                left[node_sid] = self.node_to_sid[node.children[0]]
                right[node_sid] = self.node_to_sid[node.children[1]]

        return has_children, left, right

    def get_child(self, A, has_child, side):
        # A[1, :] @ side
        csize = has_child.shape[0]
        child = np.zeros((csize,))
        for i in range(csize):
            if has_child[i]:
                child[side[i]] = A[i]

    def calculate_E_ancestral_correction(self, E, ancestors, listofnodes, p_t):
        """return row vector"""
        E_hat_c = np.dot(E, ancestors) * p_t
        return E_hat_c

    def calculate_node_weight(self, ancestors, listofnodes):
        """returns row vector"""
        w = (-np.dot(np.ones((1, self.sid_size)), ancestors)) + self.sid_size
        return w

    # @timing
    def calculate_E(self):
        """Calculate extincsion probablity"""
        listofnodes = self.listofnodes
        ancestors = self.ancestors
        left = self.left
        right = self.right
        p_s = self.p_s
        p_d = self.p_d
        p_t = self.p_t
        p_l = self.p_l
        w = self.w
        E = np.zeros((1, self.sid_size), dtype=np.double)  # TYPEDOUBLE
        for i in range(10):
            E_hat = np.dot(E, np.transpose(p_t))
            E_hat_c = self.calculate_E_ancestral_correction(
                E, ancestors, listofnodes, p_t
            )
            E = (
                p_l
                + p_s * E @ left * E @ right
                + p_d * E**2
                + E * ((E_hat - E_hat_c)) / w
            )
        return E, E_hat, E_hat_c

    def create_gid_sid_mapping(self):
        gid_sid_map = np.zeros((self.gid_size, self.sid_size), dtype=np.ubyte)
        for gid_item, sid_item in enumerate(self.gid_to_sid_map):
            gid_sid_map[gid_item, sid_item] = 1
        return gid_sid_map

    def make_gamma_vect(self, gamma_sub):
        i = sum((1 for gamma_i in gamma_sub.values() if gamma_i.children is not None))
        if i == 0:
            return np.empty((0, 0)), np.empty((0,))

        vw_gids = np.empty((i, 2), dtype=np.uint32)
        norms = np.empty((i,), dtype=np.double)
        count = 0
        for gamma_i in gamma_sub.values():
            if gamma_i.children is not None:
                vw_gids[count, 0] = self.node_to_gid[gamma_i.children[0]]
                vw_gids[count, 1] = self.node_to_gid[gamma_i.children[1]]
                norms[count] = gamma_i.norm_score
                count += 1
        return vw_gids, norms

    def calculate_P(self, E, E_hat, E_hat_c):
        """calculate probabilities"""
        gid_sid_map = self.gid_sid_map
        gamma_list = self.conf.gkey_neighbours_list
        p_s = self.p_s
        p_d = self.p_d
        p_t = self.p_t
        p_l = self.p_l
        w = self.w

        left = self.left
        right = self.right
        ancestors = self.ancestors
        P = np.zeros((self.gid_size, self.sid_size), dtype=np.double)  # TYPEDOUBLE
        P_hat = np.zeros((self.gid_size, 1), dtype=np.double)  # TYPEDOUBLE
        P_hat_c = np.zeros(
            (self.gid_size, self.sid_size), dtype=np.double
        )  # TYPEDOUBLE

        for gamma_node, gamma_sub in gamma_list:
            u_gid = self.node_to_gid[gamma_node]
            vw_gids, norms = self.make_gamma_vect(gamma_sub)
            compute_core(
                E,
                E_hat,
                E_hat_c,
                P,
                P_hat,
                P_hat_c,
                ancestors,
                u_gid,
                gid_sid_map,
                norms,
                p_d,
                p_s,
                p_t,
                left,
                right,
                vw_gids,
                w,
                self.a_has_children,
                self.a_left,
                self.a_right,
            )

        return P, P_hat, P_hat_c

    def calculate_P_np(self, E, E_hat, E_hat_c):
        """Same as calculate_P, but without the usage of numba compiled functions"""
        param = self.param
        listofnodes = self.conf.species_nodes
        gid_sid_map = self.gid_sid_map
        gamma_list = self.conf.gkey_neighbours_list
        p_s = self.p_s
        p_d = self.p_d
        p_t = self.p_t
        p_l = self.p_l
        w = self.w

        left = self.left
        right = self.right
        ancestors = self.ancestors
        E_hat_c = self.calculate_E_ancestral_correction(E, ancestors, listofnodes, p_t)
        P = np.zeros((self.gid_size, self.sid_size), dtype=np.longdouble)  # TYPEDOUBLE
        P_hat = np.zeros((self.gid_size, 1), dtype=np.longdouble)  # TYPEDOUBLE
        P_hat_c = np.zeros(
            (self.gid_size, self.sid_size), dtype=np.longdouble
        )  # TYPEDOUBLE

        for gamma_node, gamma_sub in gamma_list:
            u_gid = self.node_to_gid[gamma_node]

            for i in range(4):
                P_hat[u_gid] = np.sum(P[u_gid, :] * p_t)
                P_hat_c[u_gid, :] = np.dot(P[u_gid, :], ancestors) * p_t

                # for n in listofnodes:
                P_res = np.asmatrix(gid_sid_map[u_gid, :] * p_s)

                P_res += (
                    P[u_gid, :] * (E_hat - E_hat_c) / w
                    + E * (P_hat[u_gid] - P_hat_c[u_gid, :]) / w
                )  # ???
                P_res += p_d * 2 * P[u_gid, :] * E  # DL event #???
                P_res += p_s * (
                    np.dot(E, left) * np.dot(P[u_gid, :], right)
                    + np.dot(P[u_gid, :], left) * np.dot(E, right)
                )  # ???

                for gamma_i in gamma_sub.values():
                    if gamma_i.children is not None:
                        v_gid = self.node_to_gid[gamma_i.children[0]]
                        w_gid = self.node_to_gid[gamma_i.children[1]]
                        P_res += (
                            gamma_i.norm_score
                            * p_s
                            * (
                                np.dot(P[v_gid, :], right) * np.dot(P[w_gid, :], left)
                                + np.dot(P[w_gid, :], right) * np.dot(P[v_gid, :], left)
                            )
                        )  # S event OK
                        P_res += gamma_i.norm_score * (
                            p_d * (P[v_gid, :] * P[w_gid, :])  # D event
                            + (P_hat[w_gid] - P_hat_c[w_gid, :])
                            / w
                            * P[v_gid, :]  # T event
                            + (P_hat[v_gid] - P_hat_c[v_gid, :]) / w * P[w_gid, :]
                        )  # T event
                # if P[gamma, e_skey] < EPSILON: P[gamma, e_skey] = EPSILON
                P[u_gid, :] = P_res
                # self.print_array(P, gamma)
        return P, P_hat, P_hat_c

    def calc_score(self, E, P):
        """Calculate the probability of the global fitting"""
        root_sum = 0.0
        survive = 0.0
        conf = self.conf

        gamma_root = self.node_to_gid[conf.root_gkey]
        root_sum = np.sum(P[gamma_root, :])
        survive = np.sum(1 - E)

        return root_sum / survive

    def print_array(self, P, u_gid):
        """Print function for debugging"""
        print(len(u_gid), ": ", end="")
        for e_skey in (node.leaves_key for node in self.conf.species_nodes):
            print("[{}]".format(P[u_gid, e_skey]), end="")
        print()

    def calc_with_param(self, param_vect: Sequence[float]):
        """Calculates the probablities for a specific model parameter instance"""
        param = ModelParams(
            delta=param_vect[0], tau=param_vect[1], p_lambda=param_vect[2]
        )
        self.param = param
        if param is not None:
            row_vector = np.ones((1, self.sid_size), dtype=np.double)  # TYPEDOUBLE
            self.p_s = row_vector * (
                1 / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
            self.p_d = row_vector * (
                param.duplication_rate
                / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
            self.p_t = row_vector * (
                param.transfer_rate
                / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )
            self.p_l = row_vector * (
                param.loss_rate
                / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
            )

        self.last_ext_data = self.calculate_E()
        self.last_prob_data = self.calculate_P(*self.last_ext_data)
        self.opt_counter += 1
        res = self.calc_score(self.last_ext_data[0], self.last_prob_data[0])
        return -math.log(res)
