# axolotl/io.py
"""
Functions for handling file input and output, including loading raw data,
channel maps, and saving spike sorting results in Phy format.
"""
import numpy as np
import os
import yaml
import tempfile
import pandas as pd

def load_raw_binary(data_path: str, n_channels: int, dtype: str = 'int16') -> np.ndarray:
    """
    Loads raw binary ephys data from a file into a memory-mapped array.
    This function creates a temporary, writable copy of the data to ensure
    the original raw data file is not modified.

    Parameters
    ----------
    data_path : str
        Path to the raw binary data file (.bin or .dat).
    n_channels : int
        The number of channels in the recording.
    dtype : str
        The data type of the raw file (e.g., 'int16').

    Returns
    -------
    np.ndarray, shape (T, C)
        The raw data as a writable, memory-mapped NumPy array.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
        
    file_size_bytes = os.path.getsize(data_path)
    item_size = np.dtype(dtype).itemsize
    total_samples = file_size_bytes // (item_size * n_channels)
    
    print(f"Memory-mapping {total_samples:,} samples from {data_path}...")
    # Open the original data file as read-only
    original_data = np.memmap(data_path, dtype=dtype, mode='r', shape=(n_channels, total_samples), order='F').T
    
    # Create a temporary file to hold the writable memory-mapped copy
    # Store the temp file in the same directory to avoid cross-device issues
    temp_dir = os.path.dirname(data_path) or '.'
    temp_fp = tempfile.NamedTemporaryFile(suffix=".mmap", dir=temp_dir, delete=False)
    temp_path = temp_fp.name
    temp_fp.close() # Close the file handle so memmap can take over

    print(f"Creating a writable temporary copy at: {temp_path}")
    # Create a new, writable memory-mapped file with 'C' order for time-major access
    writable_data = np.memmap(temp_path, dtype=dtype, mode='w+', shape=original_data.shape, order='C')
    
    # Copy the data from the read-only file to the writable one
    writable_data[:] = original_data[:]
    writable_data.flush() # Ensure data is written to the temporary file
    
    print("Data mapped successfully into a writable copy.")
    return writable_data

def load_channel_map(map_path: str) -> np.ndarray:
    """
    Loads a channel map from a .npy file.

    Parameters
    ----------
    map_path : str
        Path to the .npy channel map file.

    Returns
    -------
    np.ndarray, shape (n_channels, 2)
        The electrode coordinates.
    """
    if not map_path or not os.path.exists(map_path):
        raise FileNotFoundError(f"Channel map file not found at: {map_path}")
    
    print(f"Loading channel map from {map_path}")
    return np.load(map_path)

def save_phy_results(
    output_dir: str,
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    templates: np.ndarray,
    channel_map: np.ndarray,
    config: dict
):
    """
    Saves spike sorting results in a Phy-compatible format.

    Parameters
    ----------
    output_dir : str
        The directory where results will be saved.
    spike_times : np.ndarray, shape (n_spikes,)
        The sample index of each detected spike.
    spike_clusters : np.ndarray, shape (n_spikes,)
        The cluster ID assigned to each spike.
    templates : np.ndarray, shape (n_units, n_samples, n_channels)
        The mean waveform (EI) for each unit.
    channel_map : np.ndarray
        The channel map array.
    config : dict
        The configuration dictionary used for the run.
    """
    print(f"Saving Phy-compatible results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'spike_times.npy'), spike_times.astype(np.int64))
    np.save(os.path.join(output_dir, 'spike_clusters.npy'), spike_clusters.astype(np.int32))
    np.save(os.path.join(output_dir, 'templates.npy'), templates.astype(np.float32))
    np.save(os.path.join(output_dir, 'channel_positions.npy'), channel_map)
    
    # Kilosort/Phy also expects channel_map.npy for geometry
    np.save(os.path.join(output_dir, 'channel_map.npy'), np.arange(channel_map.shape[0]))


    # Save the config file for reproducibility
    with open(os.path.join(output_dir, 'params.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print("Core Phy results saved successfully.")


def save_cluster_group_tsv(output_dir: str, cluster_groups: dict):
    """
    Saves a Phy-compatible cluster_group.tsv file.

    Parameters
    ----------
    output_dir : str
        The directory where results will be saved.
    cluster_groups : dict
        A dictionary mapping cluster IDs to their quality label (e.g., 'good', 'mua').
    """
    if not cluster_groups:
        print("No cluster groups to save. Skipping cluster_group.tsv.")
        return

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(cluster_groups, orient='index', columns=['group'])
    df.index.name = 'cluster_id'
    
    # Define the output path
    file_path = os.path.join(output_dir, 'cluster_group.tsv')
    
    # Save to a tab-separated file, as expected by Phy
    df.to_csv(file_path, sep='\t')
    print(f"Saved cluster group info to: {file_path}")
