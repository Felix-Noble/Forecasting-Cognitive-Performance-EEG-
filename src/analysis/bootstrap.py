#bootstrap.py
from src.utils.make_loggers import make_logger
from src.utils.config_loader import get_paths, get_settings_bootstrap
import multiprocessing
from pathlib import Path
import os
import numpy as np
import pandas as pd
import nolds
import psutil
from tabulate import tabulate
from numpy.random import choice
import numba
import warnings
import logging
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# def _disk_generator(filepaths: pd.Series, buffer_size: int):
#     for i in range((filepaths.shape[0] // buffer_size) - 1):
#         file_buffer = filepaths.iloc[(i*buffer_size):(i+1)*buffer_size]
#         yield file_buffer.apply(np.load)

# def _get_bootstrap_job(indices: np.ndarray):
#     """
#     Args:
#         indices: A 2D NumPy array of shape (n_channels, n_timepoints)
#                  containing integer file indices.

#     Returns:
#         A tuple containing:
#         - included_filepaths (np.ndarray): A 1D array of the unique file indices
#                                        found
#         - masks (np.ndarray): A 3D boolean array of shape
#                               (n_unique, n_channels, n_timepoints).
#                               masks[i] is the boolean mask corresponding to
#                               unique_indices[i].
#     """
    
#     filepath_axis = np.full((filepaths.shape[0],), -1)
#     included_filepaths = np.unique(indices)
   
#     filepath_axis[included_filepaths] = included_filepaths # remove filepaths not selected by np.choice
#     masks = (filepath_axis[:, None, None] == indices)

#     return masks

@numba.jit(nopython=True)
def _load_and_fill_resamples(resamples: np.ndarray, loaded_data: np.ndarray, masks):
    """Edits resample_buffer ndarray in place, adding in masked data as available"""
    channels, timepoints = masks[0,:,:].shape
    # print(channels, timepoints)
    for (data_idx) in range(masks.shape[0]):
        # print(masks[data_idx,:,:].shape)
        # print(resamples[data_idx])
        for chan in range(channels):
            for time in range(timepoints):
                if masks[data_idx, chan, time]:
                    resamples[data_idx,chan, time] = loaded_data[data_idx, chan, chan]
        # print(resamples[data_idx])

def _calculate_hurst_for_resample(resampled_data):
    """Calculates Hurst exponent for a single resampled data matrix."""
    result_chunk = np.empty((resampled_data.shape[0], *result_shape))
    for i in range(resampled_data.shape[0]):
        for j in range(resampled_data.shape[1]):
            result_chunk[i,j,:] = nolds.hurst_rs(resampled_data[i,j,:]) 
    return result_chunk

def _load_select_channel_timepoints(path):
    return np.load(path)[channels, time_range_ms[0]:time_range_ms[1]]

def _init_bootstrap_worker(w_func, 
                           f_paths: pd.Series,
                           ch_nums: np.ndarray,
                           t_range_ms: np.ndarray,
                           data_type,
                           f_shape,
                           resam_shape,
                           resul_shape):
    
    """
    Initializer for each worker process. Puts arguments into the worker's
    global namespace to avoid passing them repeatedly.
    """
    global dtype, file_shape, resample_shape, result_shape
    dtype = data_type
    file_shape = f_shape
    resample_shape = resam_shape
    result_shape = resul_shape

    global worker_func, filepaths, channels, time_range_ms
    filepaths = f_paths
    channels = np.array(ch_nums)
    time_range_ms = t_range_ms
    worker_func = w_func

    global n_timepoints 
    n_timepoints = time_range_ms[1] - time_range_ms[0]

def _bootstrap_worker(file_buffer_size, compute_buffer_size):
    """
    A worker function that generates bootstrap samples and processes with worker_func.
    """
    # from .data_generator import bootsrap_data_generator # Import within worker
    
    # data_generator = generator(filepaths, file_buffer)
    # loaded_data = np.empty(shape = (file_buffer, channels.shape[0], file_shape[1]) , dtype = dtype)
    file_idxs = [(i, min(i + file_buffer_size, filepaths.shape[0])) for i in range(0, filepaths.shape[0], file_buffer_size)]

    resamples_buffer = np.empty(shape = (compute_buffer_size, *resample_shape))
    All_masks = choice(filepaths.shape[0], size=(compute_buffer_size, channels.shape[0], n_timepoints), replace=True)
    for (i,j) in file_idxs:
            loaded_data = np.stack(filepaths.iloc[i:j].apply(_load_select_channel_timepoints), axis=0)
            _load_and_fill_resamples(resamples_buffer, loaded_data=loaded_data, masks=All_masks[i:j, :, :])
    
    # results_buffer = np.empty(shape = (compute_buffer_size, *result_shape))
    # calulate file index's and masks
    # calculate index pairs 

    # for compute_i in range(compute_buffer_size):
    #     masks = _get_bootstrap_job(choice(filepaths.shape[0], size=(channels.shape[0], n_timepoints), replace=True))

        # for (i,j) in file_idxs:
        #     loaded_data = np.stack(filepaths.iloc[i:j].apply(_load_select_channel_timepoints), axis=0)
        
            # _fill_resample(resample = resamples_buffer[compute_i, :, :], 
            #                 loaded_data = loaded_data, 
            #                 masks = masks[i:j]
            #                 )
            
    return _calculate_hurst_for_resample(resamples_buffer)

def _to_mb_str(b):
    return f"{b / (1024*1024):,.2f} MB"

def _log_progress_table(batch_num, resamples_done, n_samples_total, n_workers, file_buffer_size, compute_buffer_size, file_mem_size, compute_mem_chunk_size):
    """Generates and logs a formatted table of the current batch progress."""
    
    # Helper to format bytes into MB

    # --- Calculations for the table ---
    samples_this_batch = compute_buffer_size * n_workers
    progress_percent = (resamples_done / n_samples_total) * 100
    
    # Memory allocation calculations
    total_data_alloc = file_buffer_size * file_mem_size
    total_compute_alloc = compute_buffer_size * compute_mem_chunk_size
    per_worker_compute_alloc = total_compute_alloc / n_workers
    
    # System memory
    memory = psutil.virtual_memory()
    
    # --- Table Structure ---
    # Section for overall progress
    progress_data = [
        ["Batch Number", f"{batch_num}"],
        ["Overall Progress", f"{resamples_done:,} / {n_samples_total:,} ({progress_percent:.2f}%)"],
        ["Samples This Batch", f"{samples_this_batch:,} ({compute_buffer_size:,} per worker)"]
    ]

    # Section for memory usage
    memory_data = [
        ["Total System Available", _to_mb_str(memory.available)],
        ["Data Buffer Allocation", f"{_to_mb_str(total_data_alloc)} ({file_buffer_size} files)"],
        ["Compute Buffer Allocation", f"{_to_mb_str(total_compute_alloc)} (Total for {n_workers} workers)"],
        ["Per-Worker Compute Alloc", _to_mb_str(per_worker_compute_alloc)]
    ]
    
    # --- Generate and return the table string ---
    progress_table = tabulate(progress_data, tablefmt="fancy_grid", headers=["Metric", "Value"])
    memory_table = tabulate(memory_data, tablefmt="fancy_grid", headers=["Memory", "Size"])
    
    return f"Starting Batch {batch_num}\n{progress_table}\n{memory_table}"

def run_bootstrap_for_condition(worker_func, 
                                result_shape: tuple, 
                                trial_filepaths: pd.Series,
                                output_path, 
                                boot_settings: dict, 
                                logger, 
                                dtype=np.float64, 
                                safety_buffer=1
                                ):
    """
    Manages the bootstrap process for a single experimental condition.
    """
    if trial_filepaths.shape[0] < 1:
        logger.error("No trial files found for condition. Skipping.")
        return
    print(boot_settings)
    n_timepoints = boot_settings["time_range_ms"][1] - boot_settings["time_range_ms"][0]
    if n_timepoints < 1:
        raise ValueError(f"Time range (ms) specified poorly (n_timepoints = {n_timepoints}), check config.toml file. Current range = {boot_settings['time_range_ms'][0]} to {boot_settings['time_range_ms'][1]}")
    n_channels = len(boot_settings["channels"])
    if n_channels < 1:
        raise ValueError("No channels provided, check config.toml file")
    
    n_samples_total = boot_settings["n_samples"] 
    n_workers = boot_settings["n_workers"]

    # first_file = np.load(trial_filepaths[0])
    # file_shape = first_file.shape
    resample_shape = (n_channels, n_timepoints)

    # memory size of 1 of each procesing step across all workers 
    sample_mem_size = dtype().itemsize * (n_channels*n_timepoints) * n_workers 
    result_mem_size = dtype().itemsize * np.prod(result_shape) * n_workers 
    
    file_mem_size = dtype().itemsize * np.prod(result_shape) * n_workers 
    # TODO: add process (loading) size here along with other steps where RAM is allocated temporarily

    # del first_file
    process_buffer = 500 * (1024*1024) # allow 0.5 GB for processing to avoid hard faults 
    # size of smallest chunk needed for processing
    min_mem_chunk_size = np.sum([sample_mem_size, result_mem_size, file_mem_size]) 
    compute_mem_chunk_size = np.sum([sample_mem_size, result_mem_size])
    n_files = len(trial_filepaths) # n. files in condition directory 
    # condition_mem_size = file_mem_size * n_files # size of entire condition directory 
    init_args = (worker_func, trial_filepaths, boot_settings["channels"], boot_settings["time_range_ms"], np.float64, result_shape, resample_shape, result_shape)
    
    batch = 1
    resamples_done = 0
    print(output_path)
    with multiprocessing.Pool(processes=n_workers, initializer=_init_bootstrap_worker, initargs=init_args) as pool:
        while resamples_done < n_samples_total:
            memory = psutil.virtual_memory()
            if memory.available < min_mem_chunk_size:
                raise MemoryError("Insufficient memory available for processing, try reducing n_workers (config.toml)")
            # logger.info(f"Starting Batch {batch} -- {resamples_done}/{n_samples_total} samples done")

            available_data_chunks = (memory.available - compute_mem_chunk_size - process_buffer) // file_mem_size
            if available_data_chunks >= n_files:
                file_buffer_size = min(n_files, available_data_chunks)
            else:
                file_buffer_size = 1

            compute_buffer_size = min(n_samples_total - resamples_done, (memory.available - process_buffer - (file_buffer_size * result_mem_size)) // compute_mem_chunk_size)
            # logger.info(f"\n{tabulate([samples_per_worker], [f'Worker {n}' for n in range(1,n_workers+1)], tablefmt='grid')}")
            # Log progress in talbe
            if logger.isEnabledFor(logging.INFO):
                logger.info(_log_progress_table(batch, 
                                                resamples_done, 
                                                n_samples_total, 
                                                n_workers, 
                                                file_buffer_size, 
                                                compute_buffer_size, 
                                                file_mem_size, 
                                                compute_mem_chunk_size))
            
            # logger.info(f"Data chunks: {file_buffer_size} | alloc={(file_buffer_size * file_mem_size)/(1024*1024)}MB || Compute chunks: {compute_buffer_size} | alloc={(compute_buffer_size * compute_mem_chunk_size)/(1024*1024)}MB || Total available: {(memory.available)/(1024*1024)}MB")
            results = pool.starmap(_bootstrap_worker, ((file_buffer_size, compute_buffer_size) for _ in range(n_workers)))
            # print(len(results))
            results = np.stack(results, axis=0)
            # print(results.shape)
            results = results.squeeze()
            # print(results.shape)

            np.save(output_path / f"batch{batch}.npy", results)

            resamples_done += (results.shape[0] * results.shape[1]) # 
            batch += 1
   
def run_per_level_hurst_bootsrap_analysis():
    
    paths = get_paths()
    input_dir = Path(paths["staging_dir"]) / "PerLevel"
    output_dir = Path(paths["derivatives_dir"]) / "HurstPerLevelBootstrap"
    logger = make_logger(Path(__file__).stem)
    logger.info("Starting Bootsrap module")
    boot_settings = get_settings_bootstrap()

    for lvl in [f"0{level}"[-2:] for level in range(4,30)]:
        for result in ["C","M"]:
            fname = f"{lvl}{result}"
            trial_filepaths = pd.Series((Path(input_dir) / fname).rglob("*.npy"))
            save_file = output_dir / fname
            os.makedirs(save_file, exist_ok = True)

            logger.info(f"Starting selection: {fname}")
            try:
                
                run_bootstrap_for_condition(_calculate_hurst_for_resample, 
                                            result_shape=(len(boot_settings["channels"]), 1),
                                            trial_filepaths = trial_filepaths,
                                            output_path = save_file,
                                            boot_settings = boot_settings,
                                            logger = logger,
                                           )

            except Exception as e:
                
                logger.error(e)
            