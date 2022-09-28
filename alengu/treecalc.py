""" Computes the indexes of the loaded trees

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
import math
from collections import defaultdict
from typing import Sequence, Tuple, Mapping, Dict, Any

import numpy as np
from ete3.coretype.tree import TreeNode

from alengu.coredata import Indexes, MatrixIndexes
from alengu.dataloader import DataLoader

NUM_ERROR = 0.00001


class TreeCalculation:
    """Creates calculation over the loaded tree structure"""

    class TimeSliceBoundary:
        """Stores information about the boundaries of specific time slice"""

        def __init__(self, begin, end, height):
            self.begin = begin
            self.end = end
            self.height = height

    class TimeSliceSteps:
        """Stores steps related data from a slice"""

        def __init__(self, boundaries, slice_count, delta_t, start_times):
            self.boundaries = boundaries
            self.slice_count = slice_count
            self.delta_t = delta_t
            self.start_times = start_times

    class RankData:
        def __init__(self, rank, start, end, nodes):
            self.rank = rank
            self.start = start
            self.end = end
            self.nodes = nodes

    def create_skey_to_node_map(
        self, species_tree: TreeNode
    ) -> Mapping[frozenset, TreeNode]:
        k_to_n = dict()
        nodes = species_tree.traverse("postorder")
        for node in nodes:
            k_to_n[node.leaves_key] = node
        return k_to_n

    def calculate_indexes(self, loader: DataLoader) -> Indexes:
        indexes = Indexes()  # type: Indexes
        indexes.species_nodes = [n for n in loader.species_tree.traverse()]
        indexes.species_nodes.reverse()
        node_to_start_pos = self.create_node_to_start_pos(
            loader.species_tree
        )  # type: Mapping[TreeNode, float]
        max_start_pos = self.tree_height(node_to_start_pos)  # type: float
        indexes.rank_to_rankdata = self.create_rank_to_rankdata_map(
            node_to_start_pos, max_start_pos
        )
        indexes.rank_to_start_pos = self.rank_to_start_position_map(
            indexes.rank_to_rankdata
        )
        boundaries = self.rank_to_time_slice_boundaries_map(
            indexes.rank_to_start_pos, max_start_pos
        )
        # steps = self.rank_to_time_slice_steps(boundaries)
        indexes.rank_to_time_slice_steps = self.rank_to_time_slice_steps_closed(
            boundaries
        )

        # loop_list = self.create_loop_list(node_list, rank_start, steps_closed)
        indexes.rank_to_active_node = self.create_rank_to_active_node_map(
            indexes.rank_to_rankdata,
            [slice.boundaries for slice in indexes.rank_to_time_slice_steps],
        )
        indexes.loop_list_mod = self.create_looping_item_list_mod(
            indexes.rank_to_active_node,
            indexes.rank_to_start_pos,
            indexes.rank_to_time_slice_steps,
        )
        indexes.skey_to_ancestors = self.create_skey_to_ancestors(indexes.species_nodes)

        (
            indexes.sliceids_to_node,
            indexes.sliceid_to_len,
        ) = self.create_slice_node_map_mod(indexes.loop_list_mod)
        indexes.skey_to_node = self.create_skey_to_node_map(loader.species_tree)  #
        indexes.gkey_to_skey = loader.gene_species_map
        indexes.s_node_name = loader.s_node_name
        return indexes

    def calculate_matrix_indexes(self, loader: DataLoader, indexes: Indexes):
        m_indexes = MatrixIndexes()
        m_indexes.rank_size, m_indexes.ts_size, m_indexes.ts_len_vect = self.get_max(
            indexes.rank_to_time_slice_steps
        )
        (
            m_indexes.sid_to_node,
            m_indexes.node_to_sid,
            m_indexes.key_to_sid,
            m_indexes.sid_size,
        ) = self.make_sid_maps(loader.species_tree)

        m_indexes.act_node_list = self.create_rank_to_active_node_map(
            indexes.rank_to_rankdata,
            [slice.boundaries for slice in indexes.rank_to_time_slice_steps],
        )
        m_indexes.loop_list_mod_vect = self.create_loop_list_mod_vect(
            m_indexes.act_node_list,
            indexes.rank_to_start_pos,
            indexes.rank_to_time_slice_steps,
            m_indexes.node_to_sid,
        )
        m_indexes.loop_list_vect = self.unroll_vect(m_indexes.loop_list_mod_vect)
        # proc.create_loop_list_int(node_list, rank_start, steps, node_to_sid)
        (
            m_indexes.nodes_in_slices,
            m_indexes.len_of_slices,
        ) = self.create_slice_node_map_vect(
            m_indexes.loop_list_vect,
            m_indexes.rank_size,
            m_indexes.ts_size,
            m_indexes.sid_size,
        )
        m_indexes.delta_t_vect = self.rank_to_delta_t(indexes.rank_to_time_slice_steps)

        if loader.need_ccp:
            (
                m_indexes.gid_to_node,
                m_indexes.node_to_gid,
                m_indexes.gid_size,
            ) = self.create_gid_maps(indexes.gkey_neighbours_list)
        else:
            m_indexes.gid_to_node = loader.gid_to_node
            m_indexes.node_to_gid = loader.node_to_gid
            m_indexes.gid_size = loader.gid_size
            self.root = loader.root

        m_indexes.gid_to_sid_map = self.make_gid_to_sid_map(
            loader.gene_species_map, m_indexes.gid_to_node, m_indexes.key_to_sid
        )
        return m_indexes

    def create_node_to_dist(
        self, node: TreeNode, dist: float
    ) -> Mapping[TreeNode, float]:
        """from nodes and init distance construct node -> distance map"""
        nodes = {node: node.dist + dist}
        if len(node) > 0:
            for n_node in node.get_children():
                nodes.update(self.create_node_to_dist(n_node, node.dist + dist))
            return nodes
        else:
            return nodes

    def create_node_to_start_pos(self, node: TreeNode) -> Mapping[TreeNode, float]:
        """recursive implementation of map creation: node -> start reverse position (present is 0)"""
        if len(node.children) == 0:
            return {node: 0}
        else:  # > 0
            child_nodes = {}  # type: Dict[TreeNode, float]
            for n_node in node.get_children():
                child_nodes.update(self.create_node_to_start_pos(n_node))
            h = max((node.dist + val for node, val in child_nodes.items()))
            # nodes.update(child_nodes)
            child_nodes.update({node: h})
            return child_nodes

    #    def node_to_branch_length(self, species_tree: TreeNode) -> Mapping[TreeNode, float]:
    #        """ calculate the mapping of nodes to their height
    #            TODO: check whether its redundant"""
    #        node_height_mapping=self.node_to_strart_pos(species_tree.get_tree_root())
    #        return node_height_mapping

    def tree_height(self, node_to_branch_length: Mapping[TreeNode, float]) -> float:
        """calculate the height of the tree"""
        return max((node.dist + val for node, val in node_to_branch_length.items()))

    def create_rank_to_nodes(
        self, node_height_map, max_node_height
    ) -> Sequence[Mapping[str, Any]]:
        """list of touple ()"""
        temp = list(node_height_map.items())
        temp.sort(key=lambda v: v[1])
        s = set()
        for node, start in temp:
            s.add(start)
        starts = sorted(list(s))
        rank_to_nodes = [{"start": t, "nodes": []} for t in starts]
        for rank, nodes in enumerate(rank_to_nodes):
            for node, time in temp:
                if (
                    nodes["start"] > time or abs(nodes["start"] - time) < NUM_ERROR
                ) and (
                    nodes["start"] < (time + node.dist)
                    or abs(nodes["start"] - (time + node.dist)) < NUM_ERROR
                ):
                    nodes["nodes"].append(
                        (
                            node,
                            abs(nodes["start"] - (time + node.dist)) < NUM_ERROR,
                            abs(nodes["start"] - time) < NUM_ERROR,
                        )
                    )

        return rank_to_nodes

    def create_rank_to_rankdata_map(
        self, node_to_start_pos: Mapping[TreeNode, float], tree_height: float
    ) -> Sequence[RankData]:
        """list of touple ()"""
        temp = list(node_to_start_pos.items())  # type:Sequence[Tuple[TreeNode, float]]
        temp.sort(key=lambda v: v[1])
        f_node_list = []  # type: Sequence[self.RankData]
        for node, start in temp:
            if len(f_node_list) == 0:
                f_node_list.append(
                    self.RankData(
                        rank=0, start=tree_height - start, end=start, nodes=[node]
                    )
                )
            elif abs(f_node_list[-1].end - start) < NUM_ERROR:
                f_node_list[-1].nodes.append(node)
            else:
                f_node_list.append(
                    self.RankData(
                        rank=len(f_node_list),
                        start=tree_height - start,
                        end=start,
                        nodes=[node],
                    )
                )

        return f_node_list

    def rank_to_start_position_map(
        self, node_list: Sequence[RankData]
    ) -> Sequence[float]:
        """Calculate mapping from rank to start position; as rank runs from 0...max rank-1,
        it can be stored in a list instead Mapping"""
        return [node.end for node in node_list]

    def rank_to_time_slice_boundaries_map(
        self, rank_to_branch_start: Sequence[float], tree_height: float
    ) -> Sequence[TimeSliceBoundary]:
        """mapping of rank to time slice boundaries"""
        return [
            self.TimeSliceBoundary(start, stop, stop - start)
            for i, (start, stop) in enumerate(
                zip(rank_to_branch_start, rank_to_branch_start[1:] + [tree_height])
            )
        ]

    def rank_to_time_slice_steps(
        self, boundaries: Sequence[TimeSliceBoundary]
    ) -> Sequence[TimeSliceSteps]:
        """mapping of rank to time slice steps"""
        MIN_D = 3
        DELTA_T = 0.05
        local_D = [
            max(math.ceil(boundary.height / DELTA_T), MIN_D) for boundary in boundaries
        ]

        temp_slices_param = [
            (boundaries[i].begin, D, boundaries[i].height / D)
            for i, D in enumerate(local_D)
        ]
        slice_data = []
        for i, slice_param in enumerate(temp_slices_param):
            # print(slice_param)
            slice_data.append(
                self.TimeSliceSteps(
                    boundaries[i],
                    slice_param[1],
                    slice_param[2],
                    [
                        slice_param[0] + slice_param[2] * i
                        for i in range(slice_param[1])
                    ],
                )
            )
        return slice_data

    def rank_to_time_slice_steps_closed(
        self, boundaries: Sequence[TimeSliceBoundary]
    ) -> Sequence[TimeSliceSteps]:
        """same as rank_to_time_slice_steps, but the intervals are closed from the end of the boundary too"""
        MIN_D = 3
        DELTA_T = 0.05
        local_D = [
            max(math.ceil(boundary.height / DELTA_T), MIN_D) for boundary in boundaries
        ]

        temp_slices_param = [
            (boundaries[i].begin, D + 1, boundaries[i].height / D)
            for i, D in enumerate(local_D)
        ]
        slice_data = []
        for i, slice_param in enumerate(temp_slices_param):
            # print(slice_param)
            slice_data.append(
                self.TimeSliceSteps(
                    boundaries[i],
                    slice_param[1],
                    slice_param[2],
                    [
                        slice_param[0] + slice_param[2] * i
                        for i in range(slice_param[1])
                    ],
                )
            )
        return slice_data

    def rank_to_delta_t(self, slice_steps: Sequence[TimeSliceSteps]) -> Sequence[float]:
        """rank to delta t"""
        return [s.delta_t for s in slice_steps]

    def time_list(
        self, rank_num: int, time_slice_steps_open: Sequence[TimeSliceSteps]
    ) -> Sequence[Tuple[int, int]]:
        return [
            (rank, time_slice)
            for rank in range(rank_num)
            for time_slice in range(len(time_slice_steps_open[rank].start_times))
        ]

    def create_loop_list_int_old(
        self,
        rank_node_list,
        slices: Sequence[TimeSliceSteps],
        node_to_sid: Mapping[TreeNode, int],
    ) -> Sequence[Tuple[int, int, int, int, int, int]]:
        """TODO: does this function needed? sid -> leaves_key???"""
        rank_ts_list = self.time_list(len(rank_node_list), slices)
        loop_list = [
            (
                rank,
                ts,
                node_to_sid[node],
                node_to_sid[node.children[0]]
                if start and ts == 0 and rank > 0
                else None,
                node_to_sid[node.children[1]]
                if start and ts == 0 and rank > 0
                else None,
                i,
            )
            for rank, ts in rank_ts_list
            for i, (node, first_only, start) in enumerate(rank_node_list[rank]["nodes"])
            if (first_only and ts == 0) or not first_only
        ]
        return loop_list

    def create_loop_list_int(self, rank_node_list, ranks, slices, node_to_sid):

        prep = [
            (
                rank,
                time_slice,
                start_time,
                node_rank,
                i,
                rank_node_list[node_rank]["rank"],
                node.dist,
                rank_node_list[node_rank]["begining"],
                node,
            )
            for rank in range(len(ranks))
            for time_slice, start_time in enumerate(slices[rank].start_times)
            for node_rank in range(len(ranks))
            for i, node in enumerate(rank_node_list[node_rank]["nodes"])
        ]
        loop_list = [
            (
                rank,
                time_slice,
                node_to_sid[node],
                node_to_sid[node.children[0]]
                if (
                    node_rank > 0
                    and time_slice == 0
                    and start_time == rank_node_list[node_rank]["begining"]
                )
                else -1,
                node_to_sid[node.children[1]]
                if (
                    node_rank > 0
                    and time_slice == 0
                    and start_time == rank_node_list[node_rank]["begining"]
                )
                else -1,
                i,
            )
            for rank in range(len(ranks))
            for time_slice, start_time in enumerate(slices[rank].start_times)
            for node_rank in range(len(ranks))
            for i, node in enumerate(rank_node_list[node_rank]["nodes"])
            if rank >= rank_node_list[node_rank]["rank"]
            and node.dist + rank_node_list[node_rank]["begining"] > start_time
        ]
        return loop_list

    def create_loop_list(self, rank_node_list, ranks, slices):
        loop_list = [
            (
                start_time,
                rank,
                time_slice,
                node.leaves_key,
                node_rank > 0 and node_rank == rank and time_slice == 0,
                node,
                node.children[0] if node_rank > 0 and time_slice == 0 else None,
                node.children[1] if node_rank > 0 and time_slice == 0 else None,
            )
            for rank in range(len(ranks))
            for time_slice, start_time in enumerate(slices[rank].start_times)
            for node_rank in range(len(ranks))
            for node in rank_node_list[node_rank]["nodes"]
            if rank >= rank_node_list[node_rank]["rank"]
            and node.dist + rank_node_list[node_rank]["begining"] > start_time
        ]
        # print([l for l in loop_list if l[2] == 0])
        return loop_list

    def create_loop_list_closed(self, rank_node_list, ranks, slices_closed):
        loop_list = [
            (
                start_time,
                rank,
                time_slice,
                node.leaves_key,
                node_rank > 0 and node_rank == rank and time_slice == 0,
                node,
                node.children[0] if node_rank > 0 and time_slice == 0 else None,
                node.children[1] if node_rank > 0 and time_slice == 0 else None,
            )
            for rank in range(len(ranks))
            for time_slice, start_time in enumerate(slices_closed[rank].start_times)
            for node_rank in range(len(ranks))
            for node in rank_node_list[node_rank]["nodes"]
            if rank >= rank_node_list[node_rank]["rank"]
            and node.dist + rank_node_list[node_rank]["begining"] > start_time
        ]
        # print([l for l in loop_list if l[2] == 0])
        return loop_list

    def create_skey_to_ancestors(self, nodes):
        """Create the index to the ancestors of a node"""
        skey_to_ancestors = {}
        for node in nodes:
            node_skey = node.leaves_key
            skey_to_ancestors[node_skey] = [node]
            parent = node.up
            while parent is not None:
                skey_to_ancestors[node_skey].append(parent)
                parent = parent.up
        return skey_to_ancestors

    def create_rank_to_active_node_map(
        self,
        rank_to_rankdata: Sequence[RankData],
        rank_to_slice_boundaries: Sequence[TimeSliceBoundary],
    ) -> Sequence[Sequence[TreeNode]]:
        active_nodes = []  # type: Sequence[Sequence[TreeNode]]
        for i in range(len(rank_to_rankdata)):
            active_nodes.append([])
        for b_rank, boundaries in enumerate(rank_to_slice_boundaries):
            for rank, node_list in enumerate(rank_to_rankdata):
                for node in node_list.nodes:
                    # is node active within the boundaries i.e. node starts latter or same time as slice AND node+distance < slice end
                    node_being = rank_to_slice_boundaries[rank].begin
                    node_end = rank_to_slice_boundaries[rank].begin + node.dist
                    # most probably comparing with small epsilon range would be better
                    if (
                        node_being < boundaries.begin
                        or abs(node_being - boundaries.begin) < NUM_ERROR
                    ) and (
                        node_end > boundaries.end
                        or abs(node_end - boundaries.end) < NUM_ERROR
                    ):
                        active_nodes[b_rank].append(
                            (
                                node,
                                abs(node_being - boundaries.begin) < NUM_ERROR
                                and not abs(node_being - 0) < NUM_ERROR,
                            )
                        )
        return active_nodes

    def create_looping_item_list_mod(
        self,
        rank_to_active_node_list: Sequence[Any],
        rank_to_start_pos: Sequence[float],
        rank_to_time_slice_steps: Sequence[Sequence[int]],
    ) -> Sequence[Tuple[float, int, int, Sequence[TreeNode]]]:
        """Create item list for the main loop of modular calculation"""
        loop_list = []  # type: Sequence[Tuple[float, int, int, Sequence[TreeNode]]]
        for rank in range(len(rank_to_start_pos)):
            for time_slice, start_time in enumerate(
                rank_to_time_slice_steps[rank].start_times
            ):
                if time_slice == 0:
                    loop_list.append(
                        (start_time, rank, time_slice, rank_to_active_node_list[rank])
                    )
                else:
                    loop_list.append(
                        (
                            start_time,
                            rank,
                            time_slice,
                            [(n, False) for n, a in rank_to_active_node_list[rank]],
                        )
                    )
        return loop_list

    def convert_with_children(self, node, node_to_sid):
        f = node_to_sid[node.children[0]] if len(node.children) == 2 else -1
        g = node_to_sid[node.children[1]] if len(node.children) == 2 else -1
        return node_to_sid[node], f, g

    def create_loop_list_mod_vect(self, active_node_list, ranks, slices, node_to_sid):
        loop_list = []
        for rank in range(len(ranks)):
            for time_slice, start_time in enumerate(slices[rank].start_times):
                if time_slice == 0:
                    loop_list.append(
                        (
                            rank,
                            time_slice,
                            [
                                (*self.convert_with_children(n, node_to_sid), a)
                                for n, a in active_node_list[rank]
                            ],
                        )
                    )
                else:
                    loop_list.append(
                        (
                            rank,
                            time_slice,
                            [
                                (*self.convert_with_children(n, node_to_sid), False)
                                for n, a in active_node_list[rank]
                            ],
                        )
                    )
        return loop_list

    def unroll_vect(self, loop_list):
        unrolled = []
        for rank, time_slice, nodes in loop_list:
            for i, (e, f, g, sub) in enumerate(nodes):
                unrolled.append((rank, time_slice, e, f, g, i, sub))
        return unrolled

    def create_loop_list_mod_comp(self, rank_node_list, ranks, slices):
        loop_list = [
            (
                start_time,
                rank,
                time_slice,
                node_rank > 0 and node_rank == rank and time_slice == 0,
                [
                    node
                    for node in rank_node_list[node_rank]["nodes"]
                    if node.dist + rank_node_list[node_rank]["begining"]
                    > start_time + NUM_ERROR
                ],
            )
            for rank in range(len(ranks))
            for time_slice, start_time in enumerate(slices[rank].start_times)
            for node_rank in range(len(ranks))
            if rank >= rank_node_list[node_rank]["rank"]
        ]
        # print([l for l in loop_list if l[2] == 0])
        return loop_list

    def create_slice_node_map_mod(
        self, loop_list: Sequence[Tuple[float, int, int, Sequence[TreeNode]]]
    ) -> Tuple[
        Mapping[Tuple[int, int], Sequence[TreeNode]], Mapping[Tuple[int, int], int]
    ]:
        rank_ts_to_node_list = {
            (node[1], node[2]): [n[0] for n in node[3]] for node in loop_list
        }  # type: Mapping[Tuple[int, int], Sequence[TreeNode]]

        rank_ts_to_len_of_slices = {}  # type: Mapping[Tuple[int, int], int]
        for id, node_list in rank_ts_to_node_list.items():
            rank_ts_to_len_of_slices[id] = len(node_list)

        return rank_ts_to_node_list, rank_ts_to_len_of_slices

    def create_slice_node_map(self, loop_list):
        nodes_in_slices = defaultdict(list)
        for node in loop_list:
            nodes_in_slices[(node[1], node[2])].append(node[5])

        len_of_slices = {}
        for id, node_list in nodes_in_slices.items():
            len_of_slices[id] = len(node_list)

        return nodes_in_slices, len_of_slices

    def create_slice_node_map_vect(self, loop_list, max_rank, max_ts, max_sid):
        nodes_in_slices = np.zeros((max_rank, max_ts, max_sid), dtype=np.bool)
        len_of_slices = np.zeros((max_rank), dtype=np.int32)
        for node in loop_list:
            nodes_in_slices[node[0], node[1], node[2]] = True
            if node[1] == 0:
                len_of_slices[node[0]] += 1
        return nodes_in_slices, len_of_slices

    def make_sid_maps(self, stree: TreeNode):
        i_to_n = dict()
        n_to_i = dict()
        k_to_i = dict()
        max_i = 0
        nodes = sorted(
            stree.traverse("postorder"), key=lambda i: (len(i.leaves_key), i.leaves_key)
        )
        for i, node in enumerate(nodes):
            i_to_n[i] = node
            n_to_i[node] = i
            k_to_i[node.leaves_key] = i
            max_i = i
        return i_to_n, n_to_i, k_to_i, max_i + 1

    def get_max(
        self, slices: Sequence[TimeSliceSteps]
    ) -> Tuple[int, int, Sequence[int]]:
        max_rank = len(slices)
        ts_len = np.empty((max_rank,), dtype=np.int32)
        max_ts = 0
        for i, slice in enumerate(slices):
            ts_len[i] = slice.slice_count
            max_ts = max(max_ts, slice.slice_count)
        return max_rank, max_ts, ts_len

    def make_gid_to_sid_map(self, gene_species_map, gid_to_node, key_to_sid):
        gid_to_sid_map = []
        for i in range(len(gid_to_node)):
            if len(gid_to_node[i]) == 1:
                gid_to_sid_map.append(key_to_sid[gene_species_map[gid_to_node[i]]])
        return gid_to_sid_map

    def geneid_secies_map_calc(
        self, leaf_ids: Mapping[int, str]
    ) -> Mapping[frozenset, frozenset]:
        gene_species_map = {}
        """generate default names"""
        for name in leaf_ids.values():
            name_base = name.split("_")[0]
            gene_species_map[frozenset({name})] = frozenset({name_base})

        return gene_species_map

    def create_gid_maps(self, gamma):
        gid_to_node = dict()
        node_to_gid = dict()
        nodes = sorted([k for k, v in gamma], key=lambda i: (len(i), i))
        count = 0
        for n in nodes:
            gid_to_node[count] = n
            node_to_gid[n] = count
            count += 1
        return gid_to_node, node_to_gid, count

    def get_avg_height(self, tree):
        leaves = tree.get_leaves()
        dist = [tree.get_distance(n) for n in leaves]
        return sum(dist) / len(dist)

    def normalize_tree(self, tree):
        height = self.get_avg_height(tree)
        for i in tree.traverse():
            if i == tree:
                i.dist == 1.0
            else:
                i.dist /= height
