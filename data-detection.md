Here is the complete content for the data_and_detection.md file, combining io.py, preprocessing.py, detection.py, and config.yaml.
# Data Ingestion and Spike Detection (`data_and_detection.md`)

## 1. Configuration (`config.yaml`)

```yaml
# config.yaml
# --- Axolotl Spike Sorting Configuration ---

paths:
  # Path to the raw binary data file.
  raw_data_path: "chunk12.bin"
  # Path to the channel map/positions file.
  channel_map_path: "channel_positions.npy"
  # Directory where all results (baselines, debug files, Phy output) will be saved.
  output_dir: "output-chunk12"

recording:
  # Number of channels in the recording.
  n_channels: 512
  # Sampling rate in Hz.
  sampling_rate: 20000
  # Data type of the raw binary file ('int16', 'float32', etc.).
  dtype: 'int16'

testing:
  # Set to 'true' to enable test mode: limits data size and units found.
  enabled: false
  # If enabled, only loads the first N seconds of data.
  duration_sec: 30
  # If enabled, stops finding units after this count, even if there are more.
  max_units: 30
  # If 'enabled' is true, setting this to 'true' runs on the full recording, 
  # but still respects 'max_units' and enables debug saving.
  full_recording: true

pipeline:
  # Overall limit on the number of units to find during the peeling process.
  max_units_to_find: 5000
  # Number of samples before the spike peak (for snippet extraction). Use negative value.
  window_pre_samples: -40
  # Number of samples after the spike peak (for snippet extraction).
  window_post_samples: 80
  # Minimum time (in samples) between detected spikes on the same channel.
  refractory_samples: 10

preprocessing:
  # Length of data segments (in samples) used for baseline and noise estimation.
  segment_len: 20000

clustering:
  # --- HDBSCAN Parameters ---
  hdbscan_min_cluster_size: 75
  hdbscan_min_samples: 10
  # --- Cluster Selection/Merging Parameters (for merge_similar_clusters_extra) ---
  # Minimum P2P amplitude on a channel for it to be considered in EI comparisons.
  p2p_threshold_ei_compare: 30.0
  # Cosine similarity threshold for the most permissive merge condition (with 0 bad channels).
  cos_similarity_threshold: 0.75
  # Max residual value (negative) that constitutes a "bad channel."
  amp_residual_threshold: -20
  # Separation score threshold in PC space. Scores <= this value are allowed to merge 
  # (set high to prioritize EI similarity).
  pc_separation_threshold: 8.0

merging:
  # NEW: Minimum EI cosine similarity required to merge a new unit into an existing unit 
  # during the peeling process.
  during_peeling_merge_threshold: 0.95

2. Input/Output Utilities (io.py)
# axolotl/io.py
"""
Functions for handling file input and output, including loading raw data,
channel maps, and saving spike sorting results.
"""
import numpy as np
import h5py
import os
import yaml
import tempfile


def load_raw_binary(data_path: str, n_channels: int, dtype: str = 'int16', max_samples: int = None) -> np.ndarray:
    """
    Loads raw binary ephys data from a file into a copy-on-write memory-mapped array.
    
    This is highly efficient as it avoids making a full copy of the data on disk.
    Any modifications (like spike subtraction) are stored in memory, leaving the
    original data file untouched.

    Parameters
    ----------
    data_path : str
        Path to the raw binary data file (.bin or .dat).
    n_channels : int
        The number of channels in the recording.
    dtype : str
        The data type of the raw file (e.g., 'int16').
    max_samples : int, optional
        Maximum number of samples (time points) to load from the start of the file.
        If None, the entire file is loaded. Defaults to None.

    Returns
    -------
    np.ndarray, shape (T, C)
        The raw data as a copy-on-write, memory-mapped NumPy array.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
        
    file_size_bytes = os.path.getsize(data_path)
    item_size = np.dtype(dtype).itemsize
    
    # This check is important to prevent file corruption errors on reshape
    if file_size_bytes % (item_size * n_channels) != 0:
        print("Warning: File size is not an exact multiple of (n_channels * dtype_size).")
        print("         The file may have an incomplete final sample.")

    total_possible_samples = file_size_bytes // (item_size * n_channels)
    
    # Determine the number of samples to actually load
    if max_samples is not None and max_samples > 0:
        total_samples = min(total_possible_samples, max_samples)
        print(f"Loading a subset of {total_samples:,} samples for testing.")
    else:
        total_samples = total_possible_samples
    
    print(f"Copy-on-write memory-mapping {total_samples:,} samples from {data_path}...")
    
    # The entire logic for creating a temporary file is now replaced
    # by this single, more efficient call with mode='c'.
    writable_data = np.memmap(
        data_path, 
        dtype=dtype, 
        mode='c',  # 'c' for copy-on-write
        shape=(n_channels, total_samples), 
        order='F'
    ).T # Transpose to get (T, C) shape
    
    print("Data mapped successfully.")
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

def load_h5_results(h5_path: str) -> dict:
    """
    Loads previously sorted units from the pipeline's HDF5 output file.

    Parameters
    ----------
    h5_path : str
        Path to the results HDF5 file.

    Returns
    -------
    dict
        A dictionary where keys are unit IDs and values are dicts containing
        'spike_times', 'ei', 'selected_channels', and 'peak_channel'.
    """
    units = {}
    if not os.path.exists(h5_path):
        print(f"Warning: HDF5 results file not found at {h5_path}. Returning empty dictionary.")
        return units

    with h5py.File(h5_path, 'r') as h5:
        for unit_name in h5.keys():
            try:
                group = h5[unit_name]
                unit_id = int(unit_name.split('_')[-1])
                units[unit_id] = {
                    'spike_times': group['spike_times'][()],
                    'ei': group['ei'][()],
                    'selected_channels': group['selected_channels'][()],
                    'peak_channel': group.attrs['peak_channel']
                }
            except Exception as e:
                print(f"Could not load unit {unit_name} from HDF5 file: {e}")
    return units


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
    np.save(os.path.join(output_dir, 'channel_map.npy'), channel_map)

    # Save the config file for reproducibility
    with open(os.path.join(output_dir, 'params.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # --- Create and save cluster_group.tsv ---
    print("Creating cluster_group.tsv...")
    unique_clusters = np.unique(spike_clusters)

    with open(os.path.join(output_dir, 'cluster_group.tsv'), 'w') as f:
        f.write("cluster_id\tgroup\n")  # Write the header
        for cluster_id in unique_clusters:
            f.write(f"{cluster_id}\tgood\n")

    print("Results saved successfully.")


3. Preprocessing (preprocessing.py)
# axolotl/preprocessing.py
"""
Functions for preprocessing raw electrophysiology data, such as baseline correction.
"""

import numpy as np
from scipy.stats import trim_mean

def compute_baselines_int16_deriv_robust(raw_data, segment_len=100_000, diff_thresh=50, trim_fraction=0.05):
    """
    Compute mean baseline per channel over non-overlapping segments,
    using derivative masking + trimmed mean to suppress spike influence.

    Parameters
    ----------
    raw_data : np.ndarray, shape (T, C)
        The raw data in int16 format.
    segment_len : int
        The length of each segment in samples for baseline calculation.
    diff_thresh : int
        The derivative threshold to mask out spike-like events.
    trim_fraction : float
        The fraction of data to trim from each end before computing the mean.

    Returns
    -------
    np.ndarray, shape (C, n_segments)
        An array of baseline values in float32 format.
    """
    total_samples, n_channels = raw_data.shape
    n_segments = (total_samples + segment_len - 1) // segment_len

    baselines = np.zeros((n_channels, n_segments), dtype=np.float32)

    for seg_idx in range(n_segments):
        start = seg_idx * segment_len
        end = min(start + segment_len, total_samples)
        segment = raw_data[start:end, :]  # Shape: [S, C]

        if segment.shape[0] < 2:
            baselines[:, seg_idx] = 0
            continue

        # Compute absolute derivative
        diff_segment = np.abs(np.diff(segment, axis=0))  # Shape: [S-1, C]
        # Pad to match original length
        diff_segment = np.vstack([diff_segment, diff_segment[-1]])

        # Mask: keep only low-derivative points
        flat_mask = diff_segment < diff_thresh  # Shape: [S, C]

        # Apply mask and compute trimmed mean per channel
        for c in range(n_channels):
            flat_vals = segment[flat_mask[:, c], c].astype(np.float32)
            if len(flat_vals) > 0:
                baselines[c, seg_idx] = trim_mean(flat_vals, proportiontocut=trim_fraction)
            else:
                baselines[c, seg_idx] = 0  # Fallback

    return baselines


def subtract_segment_baselines_int16(raw_data: np.ndarray,
                                     baselines_f32: np.ndarray,
                                     segment_len: int = 100_000) -> None:
    """
    In-place baseline removal for int16 raw traces.

    Parameters
    ----------
    raw_data : np.ndarray, shape (T, C)
        The entire int16 recording in RAM. This array is modified in-place.
    baselines_f32 : np.ndarray, shape (C, n_segments)
        The baseline values from compute_baselines_int16_deriv_robust.
    segment_len : int
        The same segment length used to compute the baselines.
    """

    T, C = raw_data.shape
    C_b, n_seg = baselines_f32.shape
    if C_b != C:
        raise ValueError("Channel count mismatch between raw_data and baselines")

    # Quantise baselines once
    baselines_i16 = np.rint(baselines_f32).astype(np.int16)

    for seg_idx in range(n_seg):
        start = seg_idx * segment_len
        end   = min(start + segment_len, T) # Handle last partial segment

        # Broadcast-subtract:  [end-start, C]  -=  [C]
        raw_data[start:end, :] -= baselines_i16[:, seg_idx]

4. Detection (detection.py)
# axolotl/detection.py
"""
Functions for detecting spikes in raw data, including threshold
estimation and identifying active channels.
"""

import numpy as np
from scipy.signal import find_peaks

def find_dominant_channel_ram(
    raw_data: np.ndarray,
    positions: np.ndarray,                  # [C,2] electrode x-y (µm)
    segment_len: int = 100_000,
    n_segments: int = 10,
    peak_window: int = 30,
    top_k_neg: int = 20,
    top_k_events: int = 5,
    seed: int = 42,
    use_negative_peak: bool = False,
    top_n: int = 10,                        # how many channels to return
    min_spacing: float = 150.0              # min µm separation
) -> tuple[list[int], list[float]]:
    """
    Pick up to `top_n` channels with the largest spike-like amplitudes that are
    at least `min_spacing` µm apart.

    Returns
    -------
    top_channels   : list[int]   indices of selected electrodes
    top_amplitudes : list[float] score for each returned channel
    """
    total_samples, n_channels = raw_data.shape
    rng = np.random.default_rng(seed)

    # deterministic + random segment starts
    starts = rng.integers(0, total_samples - segment_len, size=n_segments)

    channel_amps = [[] for _ in range(n_channels)]

    for start in starts:
        seg = raw_data[start:start + segment_len, :]
        for ch in range(n_channels):
            trace = seg[:, ch].astype(np.float32)
            trace -= trace.mean()

            neg_peaks, _ = find_peaks(-trace, distance=20)
            if neg_peaks.size == 0:
                continue

            strongest = neg_peaks[np.argsort(trace[neg_peaks])[:top_k_neg]]

            for p in strongest:
                valley = trace[p]
                if use_negative_peak:
                    amp = -valley
                else:
                    w0, w1 = max(0, p - peak_window), min(segment_len, p + peak_window + 1)
                    local_max = trace[w0:w1].max()
                    amp = local_max - valley
                channel_amps[ch].append(amp)

    # mean of top-k events per channel
    mean_amp = np.zeros(n_channels, dtype=np.float32)
    for ch in range(n_channels):
        amps = np.asarray(channel_amps[ch], dtype=np.float32)
        if amps.size:
            mean_amp[ch] = np.mean(np.sort(amps)[-top_k_events:])

    # Spacing-aware greedy selection
    sorted_idx = np.argsort(mean_amp)[::-1]
    selected = []
    for idx in sorted_idx:
        if len(selected) >= top_n:
            break
        if all(np.linalg.norm(positions[idx] - positions[s]) >= min_spacing for s in selected):
            selected.append(idx)

    # If not enough well-spaced channels, pad with next best
    for idx in sorted_idx:
        if len(selected) >= top_n:
            break
        if idx not in selected:
            selected.append(idx)

    top_channels = selected
    top_amplitudes = mean_amp[top_channels].tolist()

    return top_channels, top_amplitudes


def estimate_spike_threshold_ram(
    raw_data: np.ndarray,
    ref_channel: int,
    total_samples_to_read: int,
    refractory: int = 30,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimates the spike detection threshold for a single channel.

    Returns
    -------
    threshold : float
        The calculated (negative) voltage threshold.
    spikes : np.ndarray
        Array of sample indices for detected spikes.
    next_spikes : np.ndarray
        Array of sample indices for large-amplitude events below the threshold.
    """
    trace_f = raw_data[:total_samples_to_read, ref_channel].astype(np.float32)

    neg_peaks, _ = find_peaks(-trace_f, distance=2 * refractory)
    if not len(neg_peaks):
        return 0.0, np.empty(0, dtype=int), np.empty(0, dtype=int)

    peak_vals = trace_f[neg_peaks]
    hist, edges = np.histogram(peak_vals, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2

    peak_idx, _ = find_peaks(hist)
    valley_idx, _ = find_peaks(-hist)

    threshold = None
    if len(peak_idx) > 0:
        noise_peak = peak_idx[np.argmax(hist[peak_idx])]
        cand_valleys = valley_idx[valley_idx < noise_peak]

        for v_idx in cand_valleys:
            left_peaks = peak_idx[peak_idx < v_idx]
            if not len(left_peaks):
                continue
            
            left_peak_height = hist[left_peaks[-1]]
            if hist[v_idx] < 0.25 * left_peak_height and np.sum(hist[:v_idx]) >= 200:
                threshold = centers[v_idx]
                break

    if threshold is None:
        sorted_amps = np.sort(peak_vals)
        k = min(4999, len(sorted_amps) - 1)
        threshold = sorted_amps[k] if k >= 0 else (sorted_amps[-1] if len(sorted_amps) > 0 else 0.0)

    # Find threshold crossings
    below = trace_f < threshold
    crossings = np.where(np.diff(below.astype(int)) == 1)[0] + 1
    
    spikes = []
    last_spike_time = -np.inf
    for t_cross in crossings:
        window_start = max(0, t_cross - refractory)
        window_end = min(len(trace_f), t_cross + refractory)
        local_min_idx = np.argmin(trace_f[window_start:window_end]) + window_start
        
        if local_min_idx - last_spike_time > refractory:
            spikes.append(local_min_idx)
            last_spike_time = local_min_idx

    spikes = np.unique(spikes).astype(int)

    # Cap at 50,000 spikes
    if len(spikes) > 50000:
        amps = -trace_f[spikes]
        spikes = spikes[np.argsort(amps)[-50000:]]

    # Find large sub-threshold events
    sub_mask = peak_vals >= threshold
    sub_idx = neg_peaks[sub_mask]
    if sub_idx.size > 0:
        sub_amps = -trace_f[sub_idx]
        order = np.argsort(sub_amps)[::-1]
        next_spikes = sub_idx[order[:min(500, len(order))]].astype(int)
    else:
        next_spikes = np.empty(0, dtype=int)
        
    return threshold, np.sort(spikes), next_spikes


