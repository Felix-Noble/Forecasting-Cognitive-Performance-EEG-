#bootstrap.py
from make_loggers import setup_logger
from config import get_paths, get_settings_bootsrap
import multiprocessing
from generate_data import bootsrap_data_generator
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd
import nolds
import psutil
import math

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# def hurst_multichannel(data):
#     """
#     Calculates the Hurst exponent for each channel in a multi-channel time series.

#     Args:
#         data (np.ndarray): A 2D NumPy array where each row is a time series channel.

#     Returns:
#         list: A list of Hurst exponent values, one for each channel.
#     """
#     exponents = []
#     for i in range(data.shape[0]):
#         exponents.append(nolds.hurst_rs(data[i,:]))
    
#     return exponents

def hurst_multichannel(data):
    """
    Calculates the Hurst exponent for each channel in a multi-channel time series.

    Args:
        data (np.ndarray): A 2D NumPy array where each row is a time series channel.

    Returns:
        list: A list of Hurst exponent values, one for each channel.
    """
    return tuple(nolds.hurst_rs(data[i,:]) for i in range(data.shape[0]))

def hurst_rolling_multichannel(data, window_size=1000):
    """
    Calculates the Hurst exponent for each channel across a rolling window in a multi-channel time series.

    Args:
        data (np.ndarray): A 2D NumPy array where each row is a time series channel.

    Returns:
        2D array containing rolling window hurst exponents for each channel.
    """
    rolling_exponents= []
    temp = []
    for channel in range(data.shape[0]):
        for i in range((data.shape[1]-window_size)-1):
            temp.append(nolds.hurst_rs(data[channel, (i):(i+window_size)]))

        rolling_exponents.append(temp)
        temp.clear()
    return rolling_exponents

def hurst_rolling_window_worker_func(hurst_func, window_size, *data_gen_args):

    generator = bootsrap_data_generator(*data_gen_args)
    values = []

    for data in generator:
        
        values.append(hurst_func(data, window_size))

    return values 

def hurst_worker_func(hurst_func, *data_gen_args):

    generator = bootsrap_data_generator(*data_gen_args)
    values = []

    for data in generator:
        values.append(hurst_func(data))

    return values 
    
def preload_save_hurst_worker(export_fname, *gen_args):
    resamples = tuple(bootsrap_data_generator(*gen_args))

    out = np.asarray(tuple(hurst_multichannel(data) for data in resamples))
    
    np.save(export_fname, out)
    return out.shape[0]

def main(input_dir=None, output_dir=None, dtype=np.float64):

    paths = get_paths()
    if output_dir is None:
        raise ValueError("Must specify output dir")
    
    if input_dir is None:
        input_dir = paths["staging_dir"]
    trial_filepaths = list(Path(input_dir).rglob("*.npy"))
    filepath_idxs = list(range(len(trial_filepaths)))

    boot_settings = get_settings_bootsrap()

    channels = boot_settings["channels"]
    n_channels = len(channels)

    time_range = boot_settings["time_range"]

    n_timepoints = time_range[1] - time_range[0]
    if n_timepoints < 1:
        raise ValueError(f"Must get at least 1 timepoint, got {n_timepoints}")
    
    sample_mem_size = dtype().itemsize * (n_channels*n_timepoints)
    
    N_samples = boot_settings["n_samples"]
    N_workers = boot_settings["n_workers"]
    n_samples_per_worker = N_samples // N_workers

    # Preallocate variables to avoid memory leaks

    worker_chunk_size = 0
    samples_done = 0
    available_memory_chunks = 0

    batch = 1
    done = 0

    while samples_done < N_samples:
        logger.info(f"Starting Batch {batch} -- {samples_done}/{N_samples} samples done")
        memory = psutil.virtual_memory()
        available_memory_chunks = (memory.available // sample_mem_size) - 200 # safety net of 50
        if available_memory_chunks > (N_samples - samples_done):
            available_memory_chunks = N_samples - samples_done
            samples_done = N_samples
        
        worker_chunk_size = available_memory_chunks // N_workers
        logger.info(f"\t{N_workers} workers processing {worker_chunk_size} samples each")
        
        with multiprocessing.Pool(processes=N_workers) as pool:
            results = pool.starmap(preload_save_hurst_worker, ((f"{output_dir}-batch{batch}-worker{worker_i}", worker_chunk_size, time_range, n_timepoints, channels, n_channels, trial_filepaths, filepath_idxs) for worker_i in range(N_workers)))
        
        done = sum(results)
        samples_done += done
        batch += 1

        logger.info(f"Completed {done} resamples")

        del results

    return True

if __name__ == "__main__":
    
    paths = get_paths()
    input_dir = Path(paths["staging_dir"]) / "PerLevel"
    
    output_dir = Path(paths["derivatives_dir"]) / "HurstPerLevelBootstrap"
    os.makedirs(output_dir, exist_ok = True)
    logger = setup_logger(Path(__file__).stem)
    logger.info(f"Starting Bootsrap module")

    for lvl in [f"0{l}"[-2:] for l in range(4,30)]:
        for result in ["C","M"]:
            fname = f"{lvl}{result}"
            logger.info(f"Starting selection: {fname}")
            try:

                main(input_dir / fname, output_dir / fname)

            except Exception as e:
                logger.error(e)
            

            


