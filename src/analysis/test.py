import numpy as np

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

def fetch_data(loaded_data, unique_indices, masks):
    resample = np.empty(masks[0].shape) 
    for i in range(unique_indices.shape[0]):
        resample[masks[i]] = loaded_data[unique_indices[i]][masks[i]]
    return resample

def fetch_data2(loaded_data, unique_indices, masks):
    mask_indices = np.argmax(masks, axis=0)

    # Use these indices to look up the actual file numbers.
    # This creates a 2D array of shape (n_channels, n_timepoints) that
    # is identical to the original bootstrap indices.
    source_file_indices = unique_indices[mask_indices]

    # --- Step 2: Use advanced indexing to gather data in one operation ---
    n_channels, n_timepoints = source_file_indices.shape
    channel_indices = np.arange(n_channels)[:, None]
    timepoint_indices = np.arange(n_timepoints)

    # Gather all the data in a single, highly optimized lookup.
    resample = loaded_data[source_file_indices, channel_indices, timepoint_indices]

    # --- Step 3: Handle pixels with no mask (optional but robust) ---
    # If a pixel isn't covered by any mask, argmax defaults to 0.
    # This might silently pull data from the wrong source. We can correct this
    # by finding where no masks are active and setting those pixels to 0.
    # any_mask_active = np.any(masks, axis=0)
    # resample[~any_mask_active] = 0

    return resample


samples = 5000
from time import time
n_t = 1000
n_c = 30
shape = (n_t, n_c)
preloaded_data = [np.ones(shape)*i for i in range(100)]
preloaded_data2 = np.array(preloaded_data)
indices = np.random.choice(100, shape, replace=True)
A, B = get_jobs_bootstrap(indices)

temp1 = []
t1 = time()
for _ in range(samples):
    x = fetch_data(preloaded_data, A, B)
    temp1.append(x)

print(time()-t1)
temp1 = []

t2 = time()
for _ in range(samples):
    x = fetch_data(preloaded_data2, A, B)
    temp1.append(x)

print(time()-t2)

# print(temp1[0])

# print(temp2[0])
