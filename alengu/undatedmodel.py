""" Native Python calulation of the undated model

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
import math
from typing import Mapping, Sequence
from alengu.coredata import ModelParams, Indexes

EPSILON = 2.22507e-308


class UndatedModel:
    """This is the representation of the undated computations"""

    def __init__(self, indexes: Indexes, model_params: ModelParams):
        self.indexes = indexes  # type: Indexes
        self.params = model_params  # type: ModelParams
        self.rank = indexes.rank_to_start_pos  # type: Sequence[float]
        self.slice_data = indexes.rank_to_time_slice_steps
        self.sliceids_to_node = indexes.sliceids_to_node
        self.len_of_slices = indexes.sliceid_to_len
        self.gkey_to_skey = indexes.gkey_to_skey
        self.last_ext_data = None
        self.last_prob_data = None
        self.opt_counter = 0

    def calculate_E_ancestral_correction(self, E, skey_to_ancestors, nodes, p_t):
        """Calculate ancestral correction constants for the nodes"""
        E_hat_c = {}
        for node in nodes:
            E_hat_c[node.leaves_key] = sum(
                E[anc.leaves_key] * p_t for anc in skey_to_ancestors[node.leaves_key]
            )
        return E_hat_c

    def calculate_nodes_weight(self, skey_to_ancestors, nodes):
        """Calculate node weigts"""
        w = {}
        for node in nodes:
            w[node.leaves_key] = len(nodes) - len(skey_to_ancestors[node.leaves_key])
        return w

    def calculate_E(self):
        """Calculate extinction values"""
        params = self.params
        nodes = self.indexes.species_nodes
        p_s = 1 / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        p_d = params.duplication_rate / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        p_t = params.transfer_rate / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        p_l = params.loss_rate / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )

        skey_to_ancestors = self.indexes.skey_to_ancestors
        w = self.calculate_nodes_weight(skey_to_ancestors, nodes)
        E = {}
        for node in nodes:
            E[node.leaves_key] = 0
        for i in range(10):
            E_hat = sum(E[key.leaves_key] * p_t for key in nodes)
            E_hat_c = self.calculate_E_ancestral_correction(
                E, skey_to_ancestors, nodes, p_t
            )
            for node in nodes:
                if len(node.children) == 2:
                    Ef = E[node.children[0].leaves_key]
                    Eg = E[node.children[1].leaves_key]
                else:
                    Ef = Eg = 0
                E[node.leaves_key] = (
                    p_l
                    + p_s * Ef * Eg
                    + p_d * E[node.leaves_key] ** 2
                    + E[node.leaves_key]
                    * (E_hat - E_hat_c[node.leaves_key])
                    / w[node.leaves_key]
                )
        return E, E_hat, E_hat_c

    def calculate_P(
        self, E: Mapping[str, float], E_hat: float, E_hat_c: Mapping[str, float]
    ):
        """Calculate probablilities"""
        param = self.params
        nodes = self.indexes.species_nodes
        leaves = [node for node in nodes if node.is_leaf()]
        gammas = self.indexes.gkey_neighbours_list
        p_s = 1 / (1 + param.duplication_rate + param.transfer_rate + param.loss_rate)
        p_d = param.duplication_rate / (
            1 + param.duplication_rate + param.transfer_rate + param.loss_rate
        )
        p_t = param.transfer_rate / (
            1 + param.duplication_rate + param.transfer_rate + param.loss_rate
        )
        p_l = param.loss_rate / (
            1 + param.duplication_rate + param.transfer_rate + param.loss_rate
        )
        skey_to_ancestors = self.indexes.skey_to_ancestors
        w = self.calculate_nodes_weight(skey_to_ancestors, nodes)
        P = {}
        P_hat = {}
        P_hat_c = {}

        for gamma_gkey, gamma_sub in gammas:
            for species_node in nodes:
                P[gamma_gkey, species_node.leaves_key] = 0

            for i in range(4):
                P_hat[gamma_gkey] = sum(
                    P[gamma_gkey, node.leaves_key] * p_t for node in nodes
                )
                for species_node in nodes:
                    P_hat_c[gamma_gkey, species_node.leaves_key] = sum(
                        P[gamma_gkey, anc.leaves_key] * p_t
                        for anc in skey_to_ancestors[species_node.leaves_key]
                    )

                for node in nodes:
                    e_skey = node.leaves_key
                    # leaf
                    if (
                        node in leaves
                        and gamma_gkey in self.gkey_to_skey
                        and self.gkey_to_skey[gamma_gkey] == node.leaves_key
                    ):
                        P_res = 1 * p_s
                    else:
                        P_res = 0
                    # TL
                    P_res += (
                        P[gamma_gkey, e_skey] * (E_hat - E_hat_c[e_skey]) / w[e_skey]
                        + E[e_skey]
                        * (P_hat[gamma_gkey] - P_hat_c[gamma_gkey, e_skey])
                        / w[e_skey]
                    )
                    # DL
                    P_res += p_d * 2 * P[gamma_gkey, e_skey] * E[e_skey]
                    if node not in leaves:
                        f_key = node.children[0].leaves_key
                        g_key = node.children[1].leaves_key
                        # SL
                        P_res += p_s * (
                            E[f_key] * P[gamma_gkey, g_key]
                            + P[gamma_gkey, f_key] * E[g_key]
                        )

                        for gamma_i in gamma_sub.values():
                            if gamma_i.children is not None:
                                v_gkey = gamma_i.children[0]
                                w_gkey = gamma_i.children[1]
                                # S
                                P_res += (
                                    gamma_i.norm_score
                                    * p_s
                                    * (
                                        P[v_gkey, g_key] * P[w_gkey, f_key]
                                        + P[w_gkey, g_key] * P[v_gkey, f_key]
                                    )
                                )
                                # D + T + T
                                P_res += gamma_i.norm_score * (
                                    p_d * (P[v_gkey, e_skey] * P[w_gkey, e_skey])
                                    + (P_hat[w_gkey] - P_hat_c[w_gkey, e_skey])
                                    / w[e_skey]
                                    * P[v_gkey, e_skey]
                                    + (P_hat[v_gkey] - P_hat_c[v_gkey, e_skey])
                                    / w[e_skey]
                                    * P[w_gkey, e_skey]
                                )

                    else:
                        for gamma_i in gamma_sub.values():
                            if gamma_i.children is not None:
                                v_gkey = gamma_i.children[0]
                                w_gkey = gamma_i.children[1]
                                # D + T + T
                                P_res += gamma_i.norm_score * (
                                    p_d * (P[v_gkey, e_skey] * P[w_gkey, e_skey])
                                    + (P_hat[w_gkey] - P_hat_c[w_gkey, e_skey])
                                    / w[e_skey]
                                    * P[v_gkey, e_skey]
                                    + (P_hat[v_gkey] - P_hat_c[v_gkey, e_skey])
                                    / w[e_skey]
                                    * P[w_gkey, e_skey]
                                )
                    P[gamma_gkey, e_skey] = P_res
                    # if P[gamma, e_key] < EPSILON: P[gamma, e_key] = EPSILON

        return P, P_hat, P_hat_c

    def calc_score(self, E, P):
        """Calculate the probability of the global fitting"""
        root_sum = 0.0
        survive = 0.0
        indexes = self.indexes

        gamma_root = indexes.root_gkey
        nodes = indexes.species_nodes
        for e_node in nodes:
            root_sum += P[gamma_root, e_node.leaves_key]
            survive += 1 - E[e_node.leaves_key]

        return root_sum / survive

    def print_array(self, P, gamma):
        """Print function for debugging"""
        print(len(gamma), ": ", end="")
        for s_key in (node.leaves_key for node in self.indexes.species_nodes):
            pass
            # print(f"[{P[gamma, s_key]}]", end="")
        print()

    def calc_with_param(self, param_vect: Sequence[float]):
        """Calculates the probablities for a specific model parameter instance"""
        new_params = ModelParams(
            delta=param_vect[0], tau=param_vect[1], p_lambda=param_vect[2]
        )
        self.params = new_params
        self.last_ext_data = self.calculate_E()
        self.last_prob_data = self.calculate_P(*self.last_ext_data)
        res = self.calc_score(self.last_ext_data[0], self.last_prob_data[0])
        self.opt_counter += 1
        return -math.log(res)
