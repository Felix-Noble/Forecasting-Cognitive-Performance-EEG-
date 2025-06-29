# data generator.py
import numpy as np
from numpy.random import choice
import time
from pathlib import Path

def load_data(filepaths):
    return np.array([np.load(file) for file in filepaths])

def fetch_data(loaded_data, unique_indices, masks):
    resample = np.zeros(masks[0].shape) 
    for i in range(unique_indices.shape[0]):
        resample[masks[i]] = loaded_data[unique_indices[i]][masks[i]]
    return resample

def get_jobs_bootstrap(indices: np.ndarray):
    """
    Args:
        indices: A 2D NumPy array of shape (n_channels, n_timepoints)
                 containing integer file indices.

    Returns:
        A tuple containing:
        - unique_indices (np.ndarray): A 1D array of the unique file indices
                                       found, sorted.
        - masks (np.ndarray): A 3D boolean array of shape
                              (n_unique, n_channels, n_timepoints).
                              masks[i] is the boolean mask corresponding to
                              unique_indices[i].
    """
    unique_indices = np.unique(indices)
    masks = (unique_indices[:, None, None] == indices)
    return unique_indices, masks

def bootsrap_data_generator(n_samples, time_range, n_timepoints, channels, n_channels, trial_filepaths, file_idx):
    """
    """
    n_files = len(trial_filepaths)
    
    for _ in range(n_samples):
        indices = choice(file_idx, size=(n_channels, n_timepoints), replace=True)
        print(indices) # test runs only 

        jobs = get_jobs_bootstrap(file_idx, indices)
        # print(list(jobs))
        yield load_data(trial_filepaths, jobs, indices.shape, channels, time_range)
        # then go to files in seperate func 

if __name__ == "__main__":
    import os
    import shutil
    # Test config on simulated data
    # Test is valid if bootstrap_data_generator returns array == indicies array printed with test=True
    test_trial_n = 11
    n_samples = 1
    time_range = [0,15]

    n_timepoints = time_range[1]-time_range[0]
    if n_timepoints<1:
        raise ValueError(f"Must get at least 1 timepoint, got {n_timepoints}")
    
    channels = list(range(5))
    n_channels = len(channels)
    test_dir = Path(__file__).parent / "test_dir"

    os.makedirs(test_dir, exist_ok=True)
    for i in range(test_trial_n):
        temp = np.ones((n_channels,n_timepoints))*i
        # print(temp)
        np.save(test_dir / f"{i}.npy", temp)

    trial_filepaths = list(test_dir.rglob("*.npy"))
    trial_filepaths = sorted(trial_filepaths, key=lambda f: int(Path(f).stem))
    file_idx = list(range(len(trial_filepaths)))
    print(file_idx)
    print([x.stem for x in trial_filepaths])
    
    gen = bootsrap_data_generator(
        n_samples=n_samples, 
        time_range=time_range,
        n_timepoints=n_timepoints, 
        channels=channels, 
        n_channels=len(channels),
        trial_filepaths=trial_filepaths,
        file_idx = file_idx

        )
    
    print("Starting performance benchmark...")
    t1 = time.time()
    total_bytes_processed = 0
    temp = []
    for x in gen:
        temp.append(x)
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

   
