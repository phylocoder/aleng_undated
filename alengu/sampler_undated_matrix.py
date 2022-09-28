""" Sampler for the undated Numba based implemenetation

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
from random import random
from ete3 import Tree  # TreeStyle, faces,
import numpy as np
from alengu.coredata import SamplerStat, Indexes, MatrixIndexes


EPSILON = np.finfo(np.double).eps


class TreeSamplerUndatedMatrix:
    """The sampler implementation"""

    class Step:
        """Represent a step"""

        def __init__(
            self,
            e_skey,
            u_gkey,
            event,
            val,
            f_skey=None,
            g_skey=None,
            v_gkey=None,
            w_gkey=None,
        ):
            self.e_skey = e_skey
            self.u_gkey = u_gkey
            self.event = event
            self.val = val
            self.f_skey = f_skey
            self.g_skey = g_skey
            self.v_gkey = v_gkey
            self.w_gkey = w_gkey

    def __init__(self, ext, prob, indexes: Indexes, m_indexes: MatrixIndexes, params):
        self.E_data = ext.e.reshape((-1,))
        self.E_hat_data = ext.e_hat
        self.E_hat_c_data = ext.e_hat_c
        self.P_data = prob.p
        self.P_hat_data = prob.p_hat
        self.P_hat_c_data = prob.p_hat_c
        self.root_skey = None
        self.indexes = indexes
        self.m_indexes = m_indexes
        self.gamma_map = {k: v for k, v in indexes.gkey_neighbours_list}
        self.params = params
        self.stat = SamplerStat()
        self.species_nodes = self.indexes.species_nodes
        self.species_leaves = [n for n in self.species_nodes if n.is_leaf()]
        self.p_s = 1 / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        self.p_d = params.duplication_rate / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        self.p_t = params.transfer_rate / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        self.p_l = params.loss_rate / (
            1 + params.duplication_rate + params.transfer_rate + params.loss_rate
        )
        skey_to_ancestors = self.indexes.skey_to_ancestors
        self.w = self.calculate_nodes_weight(skey_to_ancestors, indexes.species_nodes)

    def calculate_nodes_weight(self, skey_to_ancestors, nodes):
        """Calculate node weigts"""
        weights = [0] * len(self.indexes.species_nodes)
        for node in nodes:
            weights[self.m_indexes.node_to_sid[node]] = len(nodes) - len(
                skey_to_ancestors[node.leaves_key]
            )
        return weights

    def get_sample(self):
        """entry point find root"""
        start_skey = self.find_random_root()
        self.root_skey = start_skey
        tree_root = Tree(name=self.indexes.gkey_to_name[self.indexes.root_gkey])
        tree_root.dist = 0
        tree_root.events = []
        sampled_str = self.calc_sub_tree(
            tree_root, start_skey, self.indexes.root_gkey, len(self.species_nodes)
        )
        return tree_root, sampled_str

    def find_random_root(self):
        """This select the node to which the gene tree root will be mapped"""
        root_gkey = self.indexes.root_gkey
        root_gid = self.m_indexes.node_to_gid[root_gkey]

        root_sum = 0.0
        rolling_sum = 0.0
        r = random()
        root_sum = np.sum(self.P_data[root_gid, :]) + (EPSILON * self.P_data.shape[1])

        for i in range(self.P_data.shape[1]):
            rolling_sum += self.P_data[root_gid, i] + EPSILON
            if r * root_sum < rolling_sum:
                return self.m_indexes.sid_to_node[i].leaves_key
        raise ValueError("no root node selected")

    def is_leaf(self, key):
        """Checks whether a key is a leaf"""
        if len(key) == 1:
            return True
        return False

    def allowed_steps_in_tree(self, e_skey, u_gkey):
        """Generate a list of possible steps"""
        key_to_node = self.indexes.skey_to_node
        mapping = self.indexes.gkey_to_skey

        e_node = key_to_node[e_skey]
        e_sid = self.m_indexes.node_to_sid[e_node]
        if e_node.children is not None and len(e_node.children) == 2:
            f_node = e_node.children[0]
            g_node = e_node.children[1]
        # calc sum in time point
        steps = []
        u_sub = self.gamma_map[u_gkey]
        # leaf
        if self.is_leaf(u_gkey) and mapping[u_gkey] == e_skey:
            steps.append(self.Step(e_skey, u_gkey, "Leaf", 1.0))
        # else branch adds 0, so it is ignored
        u_gid = self.m_indexes.node_to_gid[u_gkey]
        if not self.is_leaf(e_skey):
            f_node = e_node.children[0]
            g_node = e_node.children[1]
            f_sid = self.m_indexes.node_to_sid[f_node]
            g_sid = self.m_indexes.node_to_sid[g_node]
            # SL events
            steps.append(
                self.Step(
                    e_skey,
                    u_gkey,
                    "SL1",
                    (
                        self.p_s * self.P_data[u_gid, f_sid] * self.E_data[g_sid]
                        + EPSILON
                    ),
                    f_skey=f_node.leaves_key,
                    g_skey=g_node.leaves_key,
                )
            )
            steps.append(
                self.Step(
                    e_skey,
                    u_gkey,
                    "SL2",
                    (
                        self.p_s * self.P_data[u_gid, g_sid] * self.E_data[f_sid]
                        + EPSILON
                    ),
                    f_skey=f_node.leaves_key,
                    g_skey=g_node.leaves_key,
                )
            )
        # DL event
        steps.append(
            self.Step(
                e_skey,
                u_gkey,
                "DL",
                (
                    self.P_data[u_gid, e_sid] * (2 * self.p_d * self.E_data[e_sid])
                    + EPSILON
                ),
            )
        )
        # TL events
        for f_node in self.species_nodes:
            if (
                f_node not in self.species_leaves
                and f_node not in self.indexes.skey_to_ancestors[e_node.leaves_key]
            ):
                f_skey = f_node.leaves_key
                f_sid = self.m_indexes.node_to_sid[f_node]

                steps.append(
                    self.Step(
                        e_skey,
                        u_gkey,
                        "TL1",
                        (
                            self.p_t
                            / self.w[f_sid]
                            * self.P_data[u_gid, f_sid]
                            * self.E_data[e_sid]
                            + EPSILON
                        ),
                        f_skey=f_skey,
                    )
                )

                steps.append(
                    self.Step(
                        e_skey,
                        u_gkey,
                        "TL2",
                        (
                            self.p_t
                            / self.w[f_sid]
                            * self.P_data[u_gid, e_sid]
                            * self.E_data[f_sid]
                            + EPSILON
                        ),
                        f_skey=f_skey,
                    )
                )
        if not self.is_leaf(u_gkey):
            for gamma_i in u_sub.values():
                if gamma_i.children is not None:
                    v_gid = self.m_indexes.node_to_gid[gamma_i.children[0]]
                    w_gid = self.m_indexes.node_to_gid[gamma_i.children[1]]
                    if not self.is_leaf(e_skey):
                        f_node = e_node.children[0]
                        g_node = e_node.children[1]
                        f_sid = self.m_indexes.node_to_sid[f_node]
                        g_sid = self.m_indexes.node_to_sid[g_node]
                        # S events
                        steps.append(
                            self.Step(
                                e_skey,
                                u_gkey,
                                "S1",
                                (
                                    gamma_i.norm_score
                                    * self.p_s
                                    * self.P_data[v_gid, f_sid]
                                    * self.P_data[w_gid, g_sid]
                                    + EPSILON
                                ),
                                f_skey=f_node.leaves_key,
                                g_skey=g_node.leaves_key,
                                v_gkey=gamma_i.children[0],
                                w_gkey=gamma_i.children[1],
                            )
                        )
                        steps.append(
                            self.Step(
                                e_skey,
                                u_gkey,
                                "S2",
                                (
                                    gamma_i.norm_score
                                    * self.p_s
                                    * self.P_data[w_gid, f_sid]
                                    * self.P_data[v_gid, g_sid]
                                    + EPSILON
                                ),
                                f_skey=f_node.leaves_key,
                                g_skey=g_node.leaves_key,
                                v_gkey=gamma_i.children[0],
                                w_gkey=gamma_i.children[1],
                            )
                        )
                    # D events
                    steps.append(
                        self.Step(
                            e_skey,
                            u_gkey,
                            "D",
                            (
                                2
                                * self.p_d
                                * gamma_i.norm_score
                                * self.P_data[v_gid, e_sid]
                                * self.P_data[w_gid, e_sid]
                                + EPSILON
                            ),
                            v_gkey=gamma_i.children[0],
                            w_gkey=gamma_i.children[1],
                        )
                    )
                    for f_node in self.species_nodes:
                        f_skey = f_node.leaves_key
                        if (
                            self.is_leaf(f_node)
                            and f_node not in self.indexes.skey_to_ancestors[e_skey]
                        ):
                            f_sid = self.m_indexes.node_to_sid[f_node]
                            # T events
                            steps.append(
                                self.Step(
                                    e_skey,
                                    u_gkey,
                                    "T1",
                                    (
                                        self.p_t
                                        / self.w[f_sid]
                                        * gamma_i.norm_score
                                        * (
                                            self.P_data[v_gid, e_sid]
                                            * self.P_data[w_gid, f_sid]
                                        )
                                        + EPSILON
                                    ),
                                    f_skey=f_skey,
                                    v_gkey=gamma_i.children[0],
                                    w_gkey=gamma_i.children[1],
                                )
                            )
                            steps.append(
                                self.Step(
                                    e_skey,
                                    u_gkey,
                                    "T2",
                                    (
                                        self.p_t
                                        / self.w[f_sid]
                                        * gamma_i.norm_score
                                        * (
                                            self.P_data[w_gid, e_sid]
                                            * self.P_data[v_gid, f_sid]
                                        )
                                        + EPSILON
                                    ),
                                    f_skey=f_skey,
                                    v_gkey=gamma_i.children[0],
                                    w_gkey=gamma_i.children[1],
                                )
                            )

        return steps

    def is_new_branch(self, key, node_list):
        for k, sub in node_list:
            if k.leaves_key == key:
                return sub
        return False

    def select_step(self, steps):
        """Select a step from the available steps"""
        steps_sum = sum((step.val for step in steps))
        r = random()
        steps_rolling = 0
        for i, step in enumerate(steps[:-1]):
            steps_rolling += step.val
            if r * steps_sum <= steps_rolling:
                return steps[i]

        return steps[-1]

    def calculate_time(self, key):
        """Looks up the time from rank/ts data"""
        rank, ts = key
        ts_step = self.indexes.rank_to_time_slice_steps[
            rank
        ]  # type: TreeCalculation.TimeSliceSteps
        time = ts_step.start_times[ts]  # type: float
        return time

    def reg_step(self, step: Step, last_event_category: str):
        """Registers the statistics of a the selected step"""
        if step.event == "Leaf":
            if step.e_skey != self.root_skey:
                self.stat.branch_counts_copies[step.e_skey] += 1
                if self.stat.branch_counts_saw[step.e_skey] == 0:
                    self.stat.branch_counts_presence[step.e_skey] += 1
                self.stat.branch_counts_saw[step.e_skey] = 1
                if last_event_category in ("S", "O"):
                    self.stat.branch_counts_singleton[step.e_skey] += 1
        elif step.event in ("S1", "S2"):
            self.stat.S += 1
            if step.e_skey != self.root_skey:
                if last_event_category in ("S", "O"):
                    self.stat.branch_counts_singleton[step.e_skey] += 1
                self.stat.branch_counts_copies[step.e_skey] += 1
                if self.stat.branch_counts_saw[step.e_skey] == 0:
                    self.stat.branch_counts_presence[step.e_skey] += 1
                self.stat.branch_counts_saw[step.e_skey] = 1

                self.stat.branch_counts_count[step.f_skey] += 1
                self.stat.branch_counts_count[step.g_skey] += 1
        elif step.event == "D":
            self.stat.D += 1
            if step.e_skey != self.root_skey:
                self.stat.branch_Ds[step.e_skey] += 1
        elif step.event in ("T1", "T2"):
            if step.e_skey != self.root_skey:
                self.stat.branch_Tfroms[step.e_skey] += 1
                self.stat.branch_Ts[step.e_skey] += 1
            self.stat.T += 1
            self.stat.T_to_from[(step.e_skey, step.f_skey)] += 1
        elif step.event in ("SL1", "SL2"):
            self.stat.S += 1
            if step.e_skey != self.root_skey:
                if last_event_category in ("S", "O"):
                    self.stat.branch_counts_singleton[step.e_skey] += 1
                self.stat.branch_counts_copies[step.e_skey] += 1
                if self.stat.branch_counts_saw[step.e_skey] == 0:
                    self.stat.branch_counts_presence[step.e_skey] += 1
                self.stat.branch_counts_saw[step.e_skey] = 1

                self.stat.branch_counts_count[step.f_skey] += 1
                self.stat.branch_counts_count[step.g_skey] += 1
            self.stat.L += 1
            if step.event == "SL1":
                if step.g_skey != self.root_skey:
                    self.stat.branch_Ls[step.g_skey] += 1
            else:
                if step.f_skey != self.root_skey:
                    self.stat.branch_Ls[step.f_skey] += 1
        elif step.event == "TL1":
            if step.e_skey != self.root_skey:
                self.stat.branch_Tfroms[step.e_skey] += 1
                self.stat.branch_Ts[step.e_skey] += 1
            self.stat.T += 1
            self.stat.T_to_from[(step.e_skey, step.f_skey)] += 1
            self.stat.L += 1
            if step.e_skey != self.root_skey:
                self.stat.branch_Ls[step.e_skey] += 1

    def get_branch_prob(self, u_gkey):
        for i_gkey, gammas in self.indexes.gkey_neighbours_list:
            if i_gkey == u_gkey:
                return sum((gamma.norm_score for gamma in gammas.values()))
        return 0

    def calc_sub_tree(self, tree, e_skey, u_gkey, last_step_category, branch_string=""):
        """Calculate a subtree of a possible gene tree - species tree mapping"""
        steps = self.allowed_steps_in_tree(e_skey, u_gkey)
        step = self.select_step(steps)
        self.reg_step(step, last_step_category)
        branch_length = self.get_branch_prob(u_gkey)
        estr = self.indexes.skey_to_node[step.e_skey].name
        gene_name = self.indexes.gkey_to_name[step.u_gkey]
        if step.event == "Leaf":
            node = tree.add_child(name=gene_name)
            node.dist = 0
            node.events = []
            sampled_str = f"{gene_name}{branch_string}:{branch_length}"
            return sampled_str
        elif step.event == "S1":
            l_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            l_sampled_tree = self.calc_sub_tree(l_node, step.f_skey, step.v_gkey, "S")
            r_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            r_sampled_tree = self.calc_sub_tree(r_node, step.g_skey, step.w_gkey, "S")
            sampled_str = f"({l_sampled_tree},{r_sampled_tree}).{estr}{branch_string}:{branch_length}"
            return sampled_str
        elif step.event == "S2":
            l_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            l_sampled_str = self.calc_sub_tree(l_node, step.g_skey, step.v_gkey, "S")
            r_node = tree.add_child(name=gene_name)
            r_node.dist = 0
            r_node.events = []
            r_sampled_str = self.calc_sub_tree(r_node, step.f_skey, step.w_gkey, "S")
            sampled_str = f"({l_sampled_str},{r_sampled_str}).{estr}{branch_string}:{branch_length}"
            return sampled_str
        elif step.event == "D":
            l_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            l_sampled_str = self.calc_sub_tree(l_node, e_skey, step.v_gkey, "D")
            r_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            r_sampled_str = self.calc_sub_tree(r_node, e_skey, step.w_gkey, "D")
            sampled_str = f"({l_sampled_str},{r_sampled_str}).D@{estr}{branch_string}:{branch_length}"
            return sampled_str
        elif step.event == "T1":
            fstr = self.indexes.skey_to_node[step.f_skey].name
            l_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            l_sampled_str = self.calc_sub_tree(l_node, e_skey, step.v_gkey, "S")
            r_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            r_sampled_str = self.calc_sub_tree(r_node, step.f_skey, step.w_gkey, "T")
            sampled_str = f"({l_sampled_str},{r_sampled_str}).T@{estr}->{fstr}{branch_string}:{branch_length}"
            return sampled_str
        elif step.event == "T2":
            fstr = self.indexes.skey_to_node[step.f_skey].name
            l_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            l_sampled_str = self.calc_sub_tree(l_node, e_skey, step.w_gkey, "S")
            r_node = tree.add_child(name=gene_name)
            l_node.dist = 0
            l_node.events = []
            r_sampled_str = self.calc_sub_tree(r_node, step.f_skey, step.v_gkey, "T")
            sampled_str = f"({l_sampled_str},{r_sampled_str}).T@{estr}->{fstr}{branch_string}:{branch_length}"
            return sampled_str
        elif step.event == "SL1":
            node = tree.add_child(name=gene_name)
            node.dist = 0
            node.events = []
            n_sampled_str = self.calc_sub_tree(
                node, step.f_skey, step.u_gkey, "S", f".{estr}{branch_string}"
            )
            return n_sampled_str
        elif step.event == "SL2":
            node = tree.add_child(name=gene_name)
            node.dist = 0
            node.events = []
            n_sampled_str = self.calc_sub_tree(
                node, step.g_skey, step.u_gkey, "S", f".{estr}{branch_string}"
            )
            return n_sampled_str
        elif step.event == "DL":
            node = tree.add_child(name=gene_name)
            node.dist = 0
            node.events = []
            n_sampled_str = self.calc_sub_tree(
                node, e_skey, step.u_gkey, "S", branch_string
            )
            return n_sampled_str
        elif step.event == "TL1":
            fstr = self.indexes.skey_to_node[step.f_skey].name
            node = tree.add_child(name=gene_name)
            node.dist = 0
            node.events = []
            n_sampled_str = self.calc_sub_tree(
                node, step.f_skey, step.u_gkey, "T", f".T@{estr}->{fstr}{branch_string}"
            )
            return n_sampled_str
        elif step.event == "TL2":
            node = tree.add_child(name=gene_name)
            node.dist = 0
            node.events = []
            n_sampled_str = self.calc_sub_tree(node, e_skey, step.u_gkey, "S")
            return n_sampled_str
        else:
            raise ValueError("ERROR")

    def get_stat(self) -> SamplerStat:
        return self.stat
