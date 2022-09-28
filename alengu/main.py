"""This is the main entry point and the command line tool of the program

For more details please consult our manuscript
"Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE
at https://github.com/ssolo/ALE.
"""
import argparse
import math
import time
from enum import Enum, unique

import numpy as np
from scipy import optimize

from alengu.coredata import (
    ModelParams,
    ExtinctionDataU,
    ProbabilityDataU,
)
from alengu.dataloader import DataLoader
from alengu.ccpcalc import CCPCalculations
from alengu.treecalc import TreeCalculation
from alengu.undatedmodel import UndatedModel
from alengu.undatedmodel_matrix import UndatedModelMatrix
from alengu.sampler_undated_matrix import TreeSamplerUndatedMatrix
from alengu.sampler_undated import TreeSamplerUndated
from alengu.timing_util import timing


@unique
class ModeTypes(Enum):
    """Enumeration to distingish between the two types of the implementation"""

    STANDARD = 0  # Standard is the pure Python implementation
    MATRIX = 1  # Matrix is using  numpy


class MainCalculations:
    """This class is responsible for the initialization and execution of the calculations"""

    def prep_calc(self, loader: DataLoader, mode: ModeTypes = ModeTypes.STANDARD):
        """Preparing index structures for computations

        This method generates the necessary lists and mappings for the computation

        """
        calc = TreeCalculation()
        indexes = calc.calculate_indexes(loader)
        ccpcalc = CCPCalculations()
        if loader.need_ccp:
            ccpcalc.add_calculations_to_indexes(loader, indexes)
        else:
            indexes.gkey_neighbours_list_b = loader.gamma_b
            indexes.gkey_neighbours_list = ccpcalc.convert_map_to_list(
                loader.gamma
            )  # type: Mapping[frozenset, Mapping[Tuple[frozenset, frozenset], Gamma]]
            indexes.root_gkey = loader.root
        indexes.gkey_to_name = indexes.create_gkey_to_name(indexes.gkey_neighbours_list)

        if mode is ModeTypes.STANDARD:
            return indexes, None
        elif mode is ModeTypes.MATRIX:
            m_indexes = calc.calculate_matrix_indexes(loader, indexes)
            return indexes, m_indexes
        else:
            raise ValueError("invalid mode {}".format(mode))

    def run(
        self,
        loader: DataLoader,
        p: ModelParams = None,
        opt: bool = True,
        native: bool = True,
        undated_model: bool = False,
        debug: bool = False,
        benchmark: bool = False,
    ):
        """This is the main entry point for computation

        This is a simplified API to run the computation. It builds the necessary objects for computation end calculate the results
        Args:
            loader: a Loader class used to load the input data
            p: parameters if we calculate a fixed position, otherwise we use the default values
            opt: whether we want to get optimized paramters or just a caluclation for a fixed paramter set (see arg p)
            native: whether run Python only computation or should use the optimized numba/C modul compiled calculations
            undated: whether we want undated calculation
            debug: do large number of additional logging
            benchmark: whether measurement of exection time is necessary
        """
        if p is None:
            opt_params = ModelParams()
        else:
            opt_params = p

        if opt:
            bound = optimize.Bounds(
                [1e-6, 1e-6, 1e-6], [10 - 1e-6, 10 - 1e-6, 10 - 1e-6]
            )

        if native:
            indexes, m_indexes = self.prep_calc(loader, mode=ModeTypes.STANDARD)
            if opt:
                undated_model = UndatedModel(indexes, None)
                opt_result = self.opt_undated(bound, undated_model)
                opt_params = ModelParams(
                    delta=opt_result.x[0],
                    tau=opt_result.x[1],
                    p_lambda=opt_result.x[2],
                )
                opt_value = -opt_result.fun
                opt_count = undated_model.opt_counter
                last_ext_data = ExtinctionDataU(*undated_model.last_ext_data)
                last_prob_data = ProbabilityDataU(*undated_model.last_prob_data)
                opt_count = undated_model.opt_counter
            else:
                undated_model = UndatedModel(indexes, opt_params)
                last_ext_data_array = undated_model.calculate_E()
                last_ext_data = ExtinctionDataU(*last_ext_data_array)
                last_prob_data_array = undated_model.calculate_P(*last_ext_data_array)
                last_prob_data = ProbabilityDataU(*last_prob_data_array)
                opt_value = math.log(
                    undated_model.calc_res(
                        last_ext_data_array[0], last_prob_data_array[0]
                    )
                )
                opt_count = 1

            sampler = TreeSamplerUndated(
                last_ext_data, last_prob_data, indexes, opt_params
            )
            res_tree = sampler.get_sample()
        else:  # matrix
            indexes, m_indexes = self.prep_calc(loader, mode=ModeTypes.MATRIX)
            if opt:
                undated_model = UndatedModelMatrix(indexes, m_indexes, None)
                opt_result = self.opt_undated(bound, undated_model)
                opt_params = ModelParams(
                    delta=opt_result.x[0],
                    tau=opt_result.x[1],
                    p_lambda=opt_result.x[2],
                )
                opt_value = -opt_result.fun
                last_ext_data = ExtinctionDataU(*undated_model.last_ext_data)
                last_prob_data = ProbabilityDataU(*undated_model.last_prob_data)
                opt_count = undated_model.opt_counter

            else:
                undated_model = UndatedModelMatrix(indexes, m_indexes, opt_params)
                last_ext_data_array = undated_model.calculate_E()
                last_ext_data = ExtinctionDataU(*last_ext_data_array)
                last_prob_data = ProbabilityDataU(
                    *undated_model.calculate_P(*last_ext_data_array)
                )
                opt_value = math.log(
                    undated_model.calc_res(last_ext_data.e, last_prob_data.p)
                )
                opt_count = 1

            sampler = TreeSamplerUndatedMatrix(
                last_ext_data, last_prob_data, indexes, m_indexes, opt_params
            )
            res_tree = sampler.get_sample()
        res_stat = sampler.stat

        return opt_value, opt_params, opt_count, res_tree, res_stat

    @timing
    def opt_undated(self, bounds, undated_model):
        """Run the optimization algorithm"""
        return optimize.minimize(
            undated_model.calc_with_param,
            np.array([0.01, 0.01, 0.01]),
            bounds=bounds,  # options={'disp': True} ,
            # tol=1e-10
        )


def create_arg_parser():
    """Build the argparese used to process the commang line args"""
    parser = argparse.ArgumentParser(description="ALE-NG")
    parser.add_argument(
        "species_tree",
        metavar="SPECIESTREE",
        type=str,
        nargs=1,
        help="file containing a single species tree (default: newick format)",
    )
    parser.add_argument(
        "gene_trees",
        metavar="GENETREES",
        type=str,
        nargs=1,
        help="file containing gene trees (default: newick format)",
    )
    parser.add_argument(
        "-aleformat", action="store_true", help="change GENETREES format to ALE"
    )
    parser.add_argument(
        "-benchmark", action="store_true", help="enables to display benchmark info"
    )
    parser.add_argument(
        "-native", action="store_true", help="use a native python computation core"
    )
    parser.add_argument(
        "-debug", action="store_true", help="enables to display debug info"
    )
    parser.add_argument(
        "-duplication_rate",
        "-delta",
        dest="duplication_rate",
        nargs="?",
        type=float,
        const=0.05,
    )
    parser.add_argument(
        "-transfer_rate",
        "-tau",
        dest="transfer_rate",
        nargs="?",
        type=float,
        const=0.05,
    )
    parser.add_argument(
        "-loss_rate", "-lambda", dest="loss_rate", nargs="?", type=float, const=0.05
    )
    parser.add_argument(
        "-speciation_rate",
        "-sigma_hat",
        dest="speciation_rate",
        nargs="?",
        type=float,
        const=1,
    )
    parser.add_argument("-undated", action="store_true")
    # sampling no
    # burn in
    # random seed
    return parser


def main():
    """Command line entry point for the ALE-NG: it parses the arguments and calles the necessary calulations"""

    parser = create_arg_parser()
    args = parser.parse_args()

    if args.benchmark:
        benchmark_start = time.time()
    loader = DataLoader()
    try:
        if args.aleformat:
            print("ALE mode")
            loader.init_from_files_ale(args.species_tree[0], args.gene_trees[0])
        else:
            loader.init_from_files(args.species_tree[0], args.gene_trees[0])
    except ValueError as v:
        print("Data load failed ({})".format(v.args[0]))
        return 1
    if args.benchmark:
        benchmark_data_loaded = time.time()
        print("Data Loaded in {}".format(benchmark_data_loaded - benchmark_start))
    calc = MainCalculations()
    opt_value, opt_params, opt_count, res_tree, res_stat = calc.run(
        loader,
        ModelParams(
            delta=args.duplication_rate, tau=args.transfer_rate, p_lambda=args.loss_rate
        ),
        opt=True,
        native=args.native,
        undated_model=args.undated,
        benchmark=args.benchmark,
        debug=args.debug,
    )
    print("Value at the end of the optimization", opt_value)
    print("The optimal parameters optimization", opt_params)
    print("Optimization steps", opt_count)
    print("Sampled tree", res_tree[0].write())
    print("ALE string", res_tree[1])
    print("Result stats", res_stat)

    # calc.run(loader, ModelParams(delta=1e-06, tau=0.0563493, p_lambda=0.0536025), opt=True, native=args.native, undated=args.undated, benchmark=args.benchmark, debug=args.debug)
    # main.main(loader, ModelParams(delta=0.01, tau=0.01, p_lambda=0.01))
    if args.benchmark:
        benchmark_finish = time.time()
        print(
            "Computation done in {}\nTotal time {}".format(
                benchmark_finish - benchmark_data_loaded,
                benchmark_finish - benchmark_start,
            )
        )


if __name__ == "__main__":
    main()
