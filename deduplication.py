# axolotl/deduplication.py
"""
Functions for removing duplicate spike detections from raw, global events.
This is a critical step in a batch-processing pipeline to ensure each
synaptic event is counted only once.
"""
import cupy as cp
import numpy as np

def deduplicate_spikes_gpu(
    raw_spike_times: cp.ndarray,
    raw_spike_channels: cp.ndarray,
    raw_data_gpu: cp.ndarray,
    refractory_samples: int,
    peak_search_samples: int = 10
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Removes duplicate spikes detected on multiple channels using a GPU-accelerated
    approach. It finds the true peak for each event and keeps only the largest
    event within a given refractory period.

    Parameters
    ----------
    raw_spike_times : cp.ndarray
        The sample indices of all detected threshold crossings.
    raw_spike_channels : cp.ndarray
        The channel index for each corresponding crossing.
    raw_data_gpu : cp.ndarray, shape (T, C)
        The entire recording, resident on the GPU.
    refractory_samples : int
        The window in samples within which to consider spikes as duplicates.
    peak_search_samples : int
        The window (in samples) around the initial detection time to search for
        the true negative peak of the waveform.

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        - unique_spike_times: The final, deduplicated spike times.
        - unique_peak_channels: The channel on which the peak amplitude was found for each unique spike.
    """
    if raw_spike_times.size == 0:
        return cp.array([], dtype=cp.int64), cp.array([], dtype=cp.int32)

    total_samples = raw_data_gpu.shape[0]

    # --- Step 1: Find true peak time and amplitude for each raw detection ---
    # Create a window of offsets to search for the peak
    search_window = cp.arange(-peak_search_samples, peak_search_samples + 1, dtype=cp.int32)
    
    # Create arrays of candidate times for peak search
    candidate_times = raw_spike_times[:, None] + search_window[None, :]
    
    # Clamp times to be within the bounds of the recording
    candidate_times = cp.clip(candidate_times, 0, total_samples - 1)
    
    # Get the waveform values at the candidate times for each spike's channel
    waveforms_at_times = raw_data_gpu[candidate_times, raw_spike_channels[:, None]]
    
    # Find the index of the minimum value in the search window for each spike
    peak_offsets_idx = cp.argmin(waveforms_at_times, axis=1)
    
    # The true peak time is the initial time plus the offset
    peak_times = candidate_times[cp.arange(len(raw_spike_times)), peak_offsets_idx]
    
    # The peak amplitude is the minimum value found
    peak_amplitudes = cp.min(waveforms_at_times, axis=1)

    # --- Step 2: Sort all events by their true peak time ---
    sort_indices = cp.argsort(peak_times)
    peak_times = peak_times[sort_indices]
    peak_amplitudes = peak_amplitudes[sort_indices]
    # The channel remains the same as the detection channel
    peak_channels = raw_spike_channels[sort_indices] 

    # --- Step 3: Identify and remove duplicates within the refractory period ---
    # Use cp.diff to find spikes that are NOT within the refractory period of the previous one
    is_not_duplicate = cp.diff(peak_times, prepend=-refractory_samples) >= refractory_samples
    
    # Create an array to mark which spikes to keep
    keep_mask = cp.ones(len(peak_times), dtype=cp.bool_)
    
    # Find indices where duplicates might occur (i.e., where is_not_duplicate is False)
    duplicate_check_indices = cp.where(~is_not_duplicate)[0]
    
    # For each potential duplicate, compare its amplitude with the previous spike
    # and mark the smaller one for removal.
    # This is an efficient way to handle pairwise comparisons in a vectorized manner.
    amplitude_current = cp.abs(peak_amplitudes[duplicate_check_indices])
    amplitude_previous = cp.abs(peak_amplitudes[duplicate_check_indices - 1])
    
    # Mark the previous spike for removal if the current one is larger
    keep_mask[duplicate_check_indices - 1] = cp.where(
        amplitude_current > amplitude_previous, False, keep_mask[duplicate_check_indices - 1]
    )
    # Mark the current spike for removal if the previous one is larger or equal
    keep_mask[duplicate_check_indices] = cp.where(
        amplitude_current <= amplitude_previous, False, keep_mask[duplicate_check_indices]
    )
    
    print(f"Deduplication: Started with {len(raw_spike_times):,} raw events, finished with {int(cp.sum(keep_mask)):,} unique spikes.")

    return peak_times[keep_mask], peak_channels[keep_mask]
