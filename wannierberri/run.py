#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------

import os
from contextlib import nullcontext
import numpy as np
from collections.abc import Iterable
from time import time
import pickle
import glob
from termcolor import cprint
import warnings
from .data_K import get_data_k
from .grid import exclude_equiv_points, Path, Grid, GridTetra
from .parallel import get_ray_cpus_count
from .result.tabresult import TABresult
from .result import ResultDict
from .restart import Checkpoint, configuration, locked_run, choose_iteration, validate_result


def print_progress(count, total, t0, tprev, progress_step_time):
    t = time() - t0
    if count == 0:
        t_remain = "unknown"
        t_est_tot = "unknown"
    else:
        t_rem_s = t / count * (total - count)
        t_remain = f"{t_rem_s:22.1f}"
        t_est_tot = f"{t_rem_s + t:22.1f}"
    if t - tprev > progress_step_time:
        print(f"{count:20d}{t:17.1f}{t_remain:>22s}{t_est_tot:>22s}", flush=True)
        tprev = t
    return tprev


def process(paralfunc,
            K_list,
            parallel,
            dump_results,
            remote_parameters,
            store_results,
            progress_step_time=300,
            progress_step_percent=1,
            on_result=None):
    t0 = time()
    t_print_prev = 0
    selK = [ik for ik, k in enumerate(K_list) if not k.was_evaluated_flag]
    numK = len(selK)
    dK_list = [K_list[ik] for ik in selK]
    if len(dK_list) == 0:
        print("nothing to process now")
        return 0, None

    print(f"processing {len(dK_list)} K points :", end=" ")
    nproc_loc = get_ray_cpus_count()
    if nproc_loc == 1:
        print("in serial.")
    else:
        print(f"using  {nproc_loc} processes.")

    print("# K-points calculated  Wall time (sec)  Est. remaining (sec)   Est. total (sec)", flush=True)
    nstep_print = max(1, nproc_loc, int(round(numK * progress_step_percent / 100)))

    def set_result(Kp, res):
        Kp.set_result(res)
        res_fac = Kp.get_result_factor()
        if dump_results:
            Kp.dump_result()
        elif not store_results:
            Kp.clear_result()
        return res_fac

    result_sum = None
    if not parallel:
        for count, Kp in enumerate(dK_list):
            res = paralfunc(Kp, **remote_parameters)
            weighted = set_result(Kp, res)
            result_sum = weighted if result_sum is None else result_sum + weighted
            if on_result is not None:
                on_result(Kp, result_sum)
            if (count + 1) % nstep_print == 0:
                t_print_prev = print_progress(count=count + 1,
                                              total=numK,
                                              t0=t0,
                                              tprev=t_print_prev,
                                              progress_step_time=progress_step_time)
    else:
        import ray
        # Bound both pending tasks and deserialized results. Mapping each ref
        # directly to its K point avoids quadratic membership scans.
        pending = {}
        source = iter(dK_list)
        window = max(1, min(4096, max(256, 2 * nproc_loc)))
        batch = max(1, min(256, nproc_loc))
        exhausted = False
        count = 0
        while pending or not exhausted:
            while not exhausted and len(pending) < window:
                try:
                    kp = next(source)
                except StopIteration:
                    exhausted = True
                    break
                pending[paralfunc.remote(kp, **remote_parameters)] = kp
            if not pending:
                break
            ready, _ = ray.wait(list(pending), num_returns=min(batch, len(pending)), timeout=1)
            if ready:
                results = ray.get(ready)
                for ref, res in zip(ready, results):
                    kp = pending.pop(ref)
                    weighted = set_result(kp, res)
                    result_sum = weighted if result_sum is None else result_sum + weighted
                    count += 1
                    if on_result is not None:
                        on_result(kp, result_sum)
                del results, res, ref
            elif on_result is not None:
                on_result(None, result_sum)
            t_print_prev = print_progress(count=count, total=numK, t0=t0,
                                          tprev=t_print_prev, progress_step_time=progress_step_time)
        print(f"Results: reused={len(K_list) - numK}, computed={count}, "
              f"written={count if dump_results else 0}", flush=True)

    t = time() - t0

    print(f"time for processing {numK:6d} K-points on {nproc_loc:3d} processes: ", end="")
    print(f"{t:10.4f} ; per K-point {t / numK:15.4f} ; proc-sec per K-point {t * nproc_loc / numK:15.4f}", flush=True)
    return len(dK_list), result_sum


@locked_run
def run(
        system,
        grid,
        calculators,
        adpt_num_iter=0,
        use_irred_kpt=True,
        symmetrize=True,
        fout_name="result",
        suffix="",
        parameters_K=None,
        file_Klist_path=None,
        restart=False,
        allow_restart=False,
        dump_results=False,
        restart_iteration=-1,
        Klist_part=10,
        parallel=True,  # will fall into serial if ray is not installed/initialized
        print_Kpoints=False,
        adpt_mesh=2,
        adpt_fac=1,
        print_progress_step_time=300,
        print_progress_step_percent=1,
        restart_recover=False,
        restart_on_corrupt="error",
        partial_save_interval=0,
        partial_output_dir=None,
):
    """
    The function to run a calculation. Substitutes the old (obsolete and removed) `integrate()` and `tabulate()`
    and allows to integrate and tabulate in one run.

    Parameters
    ----------
    system : :class:`~wannierberri.system.System`
        System under investigation
    grid : :class:`~wannierberri.Grid` or :class:`~wannierberri.Path`
        initial grid for integration. or path for tabulation
    calculators : dict
        a dictionary where keys aare any string identifiers, and the values are of :class:`~wannierberri.calculators.Calculator`
    adpt_num_iter : int
        number of recursive adaptive refinement iterations. See :ref:`sec-refine`
    adpt_mesh : int
        the size of the refinement grid (usuallay no need to change)
    adpt_fac : int
        number of K-points to be refined per quantity and criteria.
    parallel : bool
        wether to use parallelization with ray or not. if True - ray should be initialized before.
    use_irred_kpt : bool
        evaluate only symmetry-irreducible K-points
    symmetrize : bool
        symmetrize the result (always `True` if `use_irred_kpt == True`)
    fout_name : str
        beginning of the output files for each quantity after each iteration
    suffix : str
        extra marker inserted into output files to mark this particular calculation run
    print_Kpoints : bool
        print the list of K points
    file_Klist_path : str or None
        path to a directory where the K-point resolved results and factors will be stored. 
            If `None` - the directory will be `_tmp_wb` in the current working directory. 
            Nonempty directories are never removed implicitly. Use a new directory for a fresh run.
    restart_recover : bool
        Explicitly reconstruct a missing legacy K-list for regular-grid iteration 0.
        Legacy files do not prove coordinate/configuration identity; verify inputs.
    restart_on_corrupt : {"error", "recompute"}
        Reject corrupt point files, or quarantine and recompute them. Scientific
        configuration mismatches always fail. Missing files are recomputed.
    partial_save_interval : float
        Seconds between iteration-0 observation snapshots; 0 disables them.
        Enabling snapshots also enables per-point checkpoint storage.
    partial_output_dir : str or None
        Independent snapshot directory; defaults to fout_name + '.partial'.
        All generations are retained; tensors keep original integration weights.
    restart : bool
        if `True` : reads restart information from `file_Klist` and starts from there
    Klist_part : int
        write the file_Klist by portions. Increase for speed, decrease for memory saving
    parameters_K: dict
        parameters to be passed to :class:`~wannierberri.data_K.Data_K` class
    dump_results : bool
        if `True` : dumps the results of each K-point in separate files. This may be slower due 
        to read/write operations, but may save memory if the results are large (many K-points, many fermi levels/frequencies, multidimensinal tensors etc..)
    allow_restart : bool
        if `True` : allows to restart a calculation, buy storing the K-poin results in a tmp directory. If False - these files are not stored, unless 
            `dump_results` is `True` and adpt_num_iter > 0, in which case they are stored anyway, and the restart is possible,
    print_progress_step_time : float or int
        minimal intervals (in seconds) to print progress
    print_progress_step_percent : float or int
        minimal intervals (in percent) to print progress


    Returns
    --------
    dictionary of  :class:`~wannierberri.result.EnergyResult`

    Notes
    -----
    Results are also printed to ASCII files
    """
    if not np.isfinite(partial_save_interval) or partial_save_interval < 0:
        raise ValueError("partial_save_interval must be finite and nonnegative")
    if partial_save_interval:
        if isinstance(grid, Path):
            raise ValueError("Partial output supports grid iteration 0 only")
        from .partial import validate_output
        partial_output_dir = validate_output(partial_output_dir or (str(fout_name)+'.partial'),
                                             file_Klist_path or '_tmp_wb')
        allow_restart = True
    if restart_recover and not restart:
        raise ValueError("restart_recover requires restart=True")
    if restart_on_corrupt not in ("error", "recompute"):
        raise ValueError("restart_on_corrupt must be error or recompute")
    assert isinstance(parallel, bool), "parallel should be True or False"
    if parallel:
        try:
            import ray
            if not ray.is_initialized():
                warnings.warn("ray package found, but ray is not initialized, running in serial mode")
                parallel = False
        except ImportError:
            warnings.warn("ray package not found, running in serial mode")
            parallel = False



    cprint("Starting run()", 'red', attrs=['bold'])
    if parameters_K is None:
        parameters_K = {}
    print_calculators(calculators)
    # along a path only tabulating is possible
    if isinstance(grid, Path):
        print("Calculation along a path - checking calculators for compatibility")
        for key, calc in calculators.items():
            print(key, calc)
            if not calc.allow_path:
                raise ValueError(
                    f"Calculation along a Path is running, but calculator `{key}` is not compatible with a Path")
        print("All calculators are compatible")
        if symmetrize:
            print("Symmetrization switched off for Path")
            symmetrize = False
        allow_restart = False
        dump_results = False
    else:
        print("Calculation on  grid - checking calculators for compatibility")
        if use_irred_kpt:
            symmetrize = True
        for key, calc in calculators.items():
            print(key, calc)
            if not calc.allow_grid:
                raise ValueError(
                    f"Calculation on Grid is running, but calculator `{key}` is not compatible with a Grid")
        print("All calculators are compatible")

    if dump_results or restart:
        allow_restart = True
    if file_Klist_path is None:
        file_Klist_path = "_tmp_wb"
    file_Klist = os.path.join(file_Klist_path, "K_list.pickle")


    if isinstance(grid, GridTetra):
        print("Grid is tetrahedral")
    else:
        print("Grid is regular")

    print(f"The set of k points is a {grid.str_short}")

    remote_parameters = {'_system': system, '_grid': grid, '_calculators': calculators, 'symmetrize': symmetrize}
    if parallel:
        remote_parameters = {k: ray.put(v) for k, v in remote_parameters.items()}

        @ray.remote
        def paralfunc(Kpoint, _system, _grid, _calculators, symmetrize):
            # import sys
            # print("Worker sys.path:", sys.path)
            # from wannierberri.system.rvectors import Rvectors
            with get_data_k(_system, Kpoint.Kp_fullBZ, grid=_grid, Kpoint=Kpoint, **parameters_K) as data:
                resultdic = {k: v(data) for k, v in _calculators.items()}
            result = ResultDict(resultdic)
            if symmetrize:
                result = _system.pointgroup.symmetrize(result)
            return result
    else:
        def paralfunc(Kpoint, _system, _grid, _calculators, symmetrize):
            with get_data_k(_system, Kpoint.Kp_fullBZ, grid=_grid, Kpoint=Kpoint, **parameters_K) as data:
                resultdic = {k: v(data) for k, v in _calculators.items()}
            result = ResultDict(resultdic)
            if symmetrize:
                result = _system.pointgroup.symmetrize(result)
            return result

    if adpt_num_iter < 0:
        adpt_num_iter = -adpt_num_iter * np.prod(grid.div) / np.prod(adpt_mesh) / adpt_fac / 3
    adpt_num_iter = int(round(adpt_num_iter))

    if (adpt_mesh is None) or np.max(adpt_mesh) <= 1:
        adpt_num_iter = 0
    else:
        if not isinstance(adpt_mesh, Iterable):
            adpt_mesh = [adpt_mesh] * 3
        adpt_mesh = np.array(adpt_mesh)

    checkpoint = None
    if allow_restart:
        config = configuration(system, grid, calculators, parameters_K, use_irred_kpt,
                               symmetrize, adpt_mesh, adpt_fac)
        checkpoint = Checkpoint(file_Klist_path, config, calculators)
    if restart:
        start_iter, K_list, factors = checkpoint.load(grid, use_irred_kpt,
                                                     restart_iteration, restart_recover)
        result_all, restart_report = checkpoint.restore(K_list, restart_on_corrupt)
        nk_prev = len(K_list)
    else:
        if checkpoint is not None:
            checkpoint.fresh()
        K_list = grid.get_K_list(use_symmetry=use_irred_kpt)
        factors = np.array([Kp.factor for Kp in K_list])
        print("Done, sum of weights:{}".format(factors.sum()))
        start_iter = 0
        nk_prev = 0
        result_all = None

    counter = 0
    factors_old = None

    for i_iter in range(adpt_num_iter + 1):
        if checkpoint is None:
            for ik in range(nk_prev, len(K_list)):
                K_list[ik].set_storage_path(get_Kpoint_storage_path(file_Klist_path=file_Klist_path, ik=ik))
        i_iter_global = i_iter + start_iter
        print("\n" + "#" * 60)
        print(f"Iteration {i_iter_global} out of {adpt_num_iter + start_iter} ")
        if print_Kpoints:
            print("iteration {0} - {1} points. New points are:".format(i_iter + start_iter,
                                                                       len([K for K in K_list if K.result is None])))
            for i, K in enumerate(K_list):
                if not K.was_evaluated_flag:
                    print(f" K-point {i} : {K} ")
        if checkpoint is not None:
            checkpoint.begin(i_iter_global, K_list)
        if partial_save_interval and i_iter_global == 0:
            from .partial import PartialWriter
            manager = PartialWriter(partial_output_dir, checkpoint.fingerprint, K_list,
                                    partial_save_interval, restart_report if restart else None)
        else:
            manager = nullcontext(None)
        with manager as partial:
            if partial is not None and restart:
                partial.publish(result_all, strict=False)
            count_iter, result_sum_iter = process(
                paralfunc=paralfunc,
                K_list=K_list,
                parallel=parallel,
                dump_results=dump_results or allow_restart,
                store_results=allow_restart or adpt_num_iter > 0,
                progress_step_time=print_progress_step_time,
                progress_step_percent=print_progress_step_percent,
                remote_parameters=remote_parameters,
                on_result=(lambda kp, subtotal: partial.observe(kp, subtotal, result_all))
                          if partial is not None else None)
            if partial is not None:
                total = result_sum_iter if result_all is None else (
                    result_all if result_sum_iter is None else result_all+result_sum_iter)
                partial.publish(total, strict=False)

        counter += count_iter

        nk = len(K_list)
        time0 = time()

        if (result_all is None):
            result_all = result_sum_iter
        else:
            factors_old = factors
            factors = np.array([kp.factor for kp in K_list])
            factors_diff = factors[:len(factors_old)] - factors_old
            factors_diff_dict = {i: fac for i, fac in enumerate(factors_diff) if fac != 0}
            print(f"factors changed for old points : {factors_diff_dict} ")
            if result_sum_iter is not None:
                result_all += result_sum_iter
            result_all += sum(K_list[i].get_result() * fac for i, fac in factors_diff_dict.items())

        time1 = time()
        print("time1 = ", time1 - time0)
        # Recreate output on recovery as well: interruption may have happened
        # during output, or the checkpoint may have moved to another directory.
        result_all.savedata(prefix=fout_name, suffix=suffix, i_iter=i_iter + start_iter)
        if checkpoint is not None:
            checkpoint.schema = validate_result(result_all, calculators)
            checkpoint.complete(i_iter_global)

        if i_iter >= adpt_num_iter:
            break

        # Now add some more points
        Kmax = np.array([K.max for K in K_list]).T
        select_points = set().union(*(np.argsort(Km)[-adpt_fac:] for Km in Kmax))

        time2 = time()
        print("time2 = ", time2 - time1)
        l1 = len(K_list)

        nk_prev = nk

        for iK in select_points:
            K_list += K_list[iK].divide(ndiv=adpt_mesh, periodic=system.periodic, use_symmetry=use_irred_kpt)

        if use_irred_kpt and isinstance(grid, Grid):
            exclude_equiv_points(K_list, new_points=len(K_list) - l1)

        print("sum of weights now :{}".format(sum(Kp.factor for Kp in K_list)))

    print(f"Totally processed {counter} K-points ")
    print("run() finished")

    # This reordering is important for the case of Path calculation in parallel
    # Because in this case the K-points are not processed in the order of the path
    for key, val in result_all.results.items():
        if isinstance(val, TABresult):
            if isinstance(grid, Path):
                val.self_to_path(path=grid)
            elif isinstance(grid, Grid):
                val.self_to_grid()

    return result_all


def print_calculators(calculators):
    cprint("Using the follwing calculators : \n" + "#" * 60 + "\n", "cyan", attrs=["bold"])
    for key, val in calculators.items():
        cprint(f" '{key}' ", "magenta", attrs=["bold"], end="")
        print(" : ", end="")
        cprint(f" {val} ", "yellow", attrs=["bold"], end="")
        print(f" : {val.comment}")
    cprint("#" * 60, "cyan", attrs=["bold"])


def get_Kpoint_storage_path(file_Klist_path, ik):
    return os.path.join(file_Klist_path, f"_Kp-{ik}.pickle")


def write_factors(file_Klist_path, factors, iter):
    with open(os.path.join(file_Klist_path, f"factors_iter-{iter:08d}.npy"), 'wb') as f:
        np.save(f, factors)


def read_factors(file_Klist_path, iter):
    files = glob.glob(os.path.join(file_Klist_path, "factors_iter-*.npy"))
    indices = [int(os.path.basename(f).split("-")[-1].split(".")[0]) for f in files]
    index = choose_iteration(indices, iter)
    with open(os.path.join(file_Klist_path, f"factors_iter-{index:08d}.npy"), "rb") as f:
        return index, np.load(f, allow_pickle=False)
