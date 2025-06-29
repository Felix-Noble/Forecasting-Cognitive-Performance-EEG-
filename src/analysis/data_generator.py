# data generator.py
import numpy as np
from numpy.random import choice
import time
import glob
from pathlib import Path

# def within_subject_load_npy(trial_filepaths, n_timepoints, time_idx, indexes,channel, 
#                             lock, delay=0.1):
#     out = np.zeros((n_timepoints,1))
    
#     for i in time_idx:
#         lock.aquire()
#         temp = np.load(trial_filepaths[indexes[i]])
#         out[i] = temp[channel, i]
#         lock.release()
#         time.sleep(delay)
    
#     return out


# def make_jobs_bootstrap(n_samples, channels, trial_filepaths, n_timepoints):
#     trial_indexes = tuple(range(len(trial_filepaths)))
#     time_idx = list(range(n_timepoints)) # placeholder

#     for _ in range(n_samples):
#         for ch in channels:
#             indexs = np.random.choice(trial_indexes,size=trial_indexes.size, replace=True)
#             yield (trial_filepaths, n_timepoints, time_idx, indexs, ch)

def fetch_data(trial_filepaths, jobs, shape, channels, time_range):
    out = np.zeros(shape)
    for job in jobs:
        temp = np.load(trial_filepaths[job[0]])[channels, time_range[0]:time_range[1]+1]
    
        for idx in job[1:]:
            out[idx[0],idx[1]] = temp[idx[0],idx[1]]

    return out 

def get_jobs(file_idxs, indices):
    """
    """
    return (
        (i, *vals)
        for i in file_idxs
        if (vals := np.argwhere(indices==i)).size > 0
    )


def bootsrap_data_generator(n_samples, time_range, n_timepoints, channels, n_channels, trial_filepaths, file_idxs):
    """
    """
    
    for _ in range(n_samples):
        indices = choice(file_idxs, size=(n_channels, n_timepoints), replace=True)
        # print(indices) # test runs only 

        jobs = get_jobs(file_idxs, indices)
        yield fetch_data(trial_filepaths, jobs, indices.shape, channels, time_range)
        # then go to files in seperate func 

def generator(worker, job_list): 
    for job in job_list:
        
        yield worker(*job)

def foo(x,y):
    return x

if __name__ == "__main__":
    import os
    # Test config on simulated data
    # Test is valid if bootstrap_data_generator returns array == indicies array printed with test=True
    
    time_range = [0,100]
    n_timepoints = time_range[1]-time_range[0]
    if n_timepoints<1:
        raise ValueError(f"Must get at least 1 timepoint, got {n_timepoints}")
    
    channels = list(range(30))
    n_channels = len(channels)
    test_dir = Path(os.getcwd())/"test_dir"
    os.makedirs(test_dir, exist_ok=True)
    test_n = 5
    for i in range(test_n):
        temp = np.ones((n_channels,n_timepoints))*i
        np.save(test_dir / f"{i}.npy", temp)

    trial_filepaths = list((test_dir).rglob("*.npy"))

    n_samples = 1

    gen = bootsrap_data_generator(
        n_samples=n_samples, 
        time_range=time_range,
        n_timepoints=n_timepoints, 
        channels=channels, 
        trial_filepaths=trial_filepaths,
        test_run=True)
    
    print("Starting performance benchmark...")
    t1 = time.time()
    total_bytes_processed = 0
    for x in gen:
        print(x)
        total_bytes_processed += x.nbytes

    # --- Calculations for Display ---
    t2 = time.time()
    elapsed_time = t2 - t1
    mb_processed = total_bytes_processed / (1024 * 1024) # More accurate MB
    rate_mbs = mb_processed / elapsed_time if elapsed_time > 0 else 0
    num_trials = len(trial_filepaths)
    num_channels = len(channels)

    # --- ASCII Report Generation ---
    # This creates a clean, aligned, and visually appealing summary.
    title = " PERFORMANCE BENCHMARK RESULTS "
    width = 60 # You can adjust the width of the box

    print("\n" + "#" * width)
    print(f"#{title.center(width - 2)}#")
    print("#" * width)

    print(f"| {'Configuration':<{width-3}} |")
    print(f"|{'-' * (width-2)}|")
    print(f"| {'Samples Generated:':<30} {n_samples:<{width-34}} |")
    print(f"| {'Timepoints per Sample:':<30} {n_timepoints:<{width-34}} |")
    print(f"| {'Number of Channels:':<30} {num_channels:<{width-34}} |")
    print(f"| {'Number of Input Trials:':<30} {num_trials:<{width-34}} |")
    print(f"|{' ' * (width-2)}|")

    print(f"| {'Performance':<{width-3}} |")
    print(f"|{'-' * (width-2)}|")
    print(f"| {'Total Time Elapsed:':<30} {elapsed_time:.2f} seconds{'':<{width-45}} |")
    print(f"| {'Total Data Processed:':<30} {mb_processed:.2f} MB{'':<{width-40}} |")
    print(f"| {'Processing Rate:':<30} {rate_mbs:.2f} MB/s{'':<{width-41}} |")
    print("#" * width + "\n")

