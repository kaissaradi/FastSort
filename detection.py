# axolotl/detection.py
"""
Functions for detecting spikes in raw data, including threshold
estimation and identifying active channels. This file includes both legacy
CPU-based methods and new GPU-accelerated functions for global detection.
"""

import numpy as np
import cupy as cp
from scipy.signal import find_peaks

# --------------------------------------------------------------------------
# --- New GPU-Accelerated Functions for the Batch Pipeline ---
# --------------------------------------------------------------------------

def detect_spikes_globally_gpu(
    raw_data_gpu: cp.ndarray,
    sampling_rate: int,
    thresholds_gpu: cp.ndarray,
    peak_search_samples: int = 10
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Detects all spike events across all channels simultaneously on the GPU.

    This function performs a highly parallel search for threshold crossings
    and is the first major step in the batch-processing pipeline.

    Parameters
    ----------
    raw_data_gpu : cp.ndarray, shape (T, C)
        The entire recording, resident on the GPU.
    sampling_rate : int
        The sampling rate of the recording in Hz.
    thresholds_gpu : cp.ndarray, shape (C,)
        A 1D array of negative detection thresholds for each channel.
    peak_search_samples : int
        The window (in samples) before and after a threshold crossing to
        search for the true negative peak.

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        - raw_spike_times: A 1D CuPy array of sample indices for all detected events.
        - raw_spike_channels: A 1D CuPy array of the channel index for each event.
    """
    print("Starting global spike detection on GPU...")
    
    # --- Step 1: Find all threshold crossings ---
    # Create a boolean mask where the data is below the threshold
    below_threshold = raw_data_gpu < thresholds_gpu[cp.newaxis, :]
    
    # Find the exact sample where the signal crosses the threshold by looking for
    # a change from False to True. The `diff` operation is perfect for this.
    crossings = cp.where(cp.diff(below_threshold, axis=0, prepend=False))
    
    # `crossings` is a tuple of (row_indices, col_indices).
    # We add 1 to the row_indices because diff gives the start of the transition.
    crossing_times, crossing_channels = crossings[0] + 1, crossings[1]
    
    print(f"Found {len(crossing_times):,} raw threshold crossings.")
    
    # The deduplication step will handle finding the true peak and removing duplicates,
    # so we can return these raw crossing events directly.
    return crossing_times.astype(cp.int64), crossing_channels.astype(cp.int32)


# --------------------------------------------------------------------------
# --- Legacy CPU-Based Functions (Kept for Reference) ---
# --------------------------------------------------------------------------

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
