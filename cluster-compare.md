Got it. Combining clustering.py, subtraction.py, and comparison.py into a single Markdown file named unit_processing_and_comparison.md, ensuring each original file's content is enclosed in its own Python code block.
Unit Processing and Comparison (unit_processing_and_comparison.md)
1. Electrical Image (EI) Comparison (comparison.py)
This module provides functions for quantifying the similarity between Electrical Images (EIs), which is crucial for unit merging and quality control. The primary comparison uses cosine similarity after a lag-tolerant alignment based on cross-correlation on the dominant channel. It also includes utilities for sub-sample alignment and subtraction-based comparison.
# axolotl/comparison.py
"""
Functions for comparing electrical images (EIs) using similarity metrics
like cosine similarity, lag-tolerant alignment, and subtraction.
"""
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d

def compare_eis(eis, ei_template=None, max_lag=30, thr=30.0):
    """
    If `ei_template` is None  → full pair-wise similarity matrix  [k,k].
    If `ei_template` given     → column vector of similarities  [k,1].
    Similarity is cosine over signal channels only, after lag-tolerant alignment.
    
    Parameters
    ----------
    eis : list or np.ndarray, shape (k, C, T)
        A list or stack of EIs to compare.
    ei_template : np.ndarray, shape (C, T), optional
        A single template to compare all EIs against. If None, performs pairwise comparison.
    max_lag : int
        Maximum sample shift allowed for alignment based on the dominant trace.
    thr : float
        P2P amplitude threshold for selecting "signal channels" to include in the cosine calculation.

    Returns
    -------
    np.ndarray
        Similarity matrix or vector.
    """
    if isinstance(eis, list):
        eis = np.stack(eis, axis=0)
    k, C, T = eis.shape
    ptp = np.ptp(eis, axis=2)

    if ei_template is None:
        # --- Pair-wise comparison (A vs B) ---
        sim = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            ei_i = eis[i]
            dom_i = np.argmax(ptp[i])
            trace_i = ei_i[dom_i]
            for j in range(i, k):
                ei_j = eis[j]
                dom_j = np.argmax(ptp[j])
                trace_j = ei_j[dom_j]
                
                # Align traces by max cross-correlation in the max_lag window
                lags = np.arange(-T + 1, T) # full range
                xc = correlate(trace_i, trace_j, mode="full", method="auto")
                center = len(xc) // 2
                
                # Search only in the range corresponding to [-max_lag, max_lag]
                lag_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
                
                # Find the best shift (lag) to apply to ei_j to match ei_i
                shift = lags[lag_indices[np.argmax(xc[lag_indices])]]
                ei_j_aligned = np.roll(ei_j, shift, axis=1)
                
                # Use channels where EITHER template is strong
                mask = (ptp[i] > thr) | (ptp[j] > thr)
                if mask.any():
                    a = ei_i[mask].ravel()
                    b = ei_j_aligned[mask].ravel()
                    # Cosine similarity
                    val = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                else:
                    val = 0.0
                sim[i, j] = sim[j, i] = val
        return sim
    else:
        # --- Comparison against a single template (A vs Template) ---
        sim = np.zeros((k, 1), dtype=np.float32)
        ptp_t = np.ptp(ei_template, axis=1)
        dom_t = int(np.argmax(ptp_t))
        trace_t = ei_template[dom_t]
        
        for i in range(k):
            ei_i = eis[i]
            dom_i = int(np.argmax(ptp[i]))
            trace_i = ei_i[dom_i]
            
            # Align trace_i to trace_t
            lags = np.arange(-T + 1, T)
            xc = correlate(trace_i, trace_t, mode="full", method="auto")
            
            lag_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
            # Find the best shift (lag) to apply to ei_template to match ei_i
            shift = lags[lag_indices[np.argmax(xc[lag_indices])]]

            ei_t_aligned = np.roll(ei_template, shift, axis=1)
            
            mask = (ptp[i] > thr) | (ptp_t > thr)
            if mask.any():
                a = ei_i[mask].ravel()
                b = ei_t_aligned[mask].ravel()
                sim[i, 0] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            else:
                sim[i, 0] = 0.0
        return sim


def sub_sample_align_ei(ei_template, ei_candidate, ref_channel, upsample=10, max_shift=2.0):
    """
    Align ei_candidate to ei_template using sub-sample alignment on the reference channel.
    Uses cubic interpolation and high-resolution cross-correlation.
    """
    C, T = ei_template.shape
    assert ei_candidate.shape == (C, T), "Shape mismatch"

    t = np.arange(T)
    t_interp = np.linspace(0, T - 1, T * upsample)

    # Interpolate reference traces
    interp_template = interp1d(t, ei_template[ref_channel], kind='cubic', bounds_error=False, fill_value=0.0)
    interp_candidate = interp1d(t, ei_candidate[ref_channel], kind='cubic', bounds_error=False, fill_value=0.0)

    template_highres = interp_template(t_interp)
    candidate_highres = interp_candidate(t_interp)

    # High-res cross-correlation
    full_corr = correlate(candidate_highres, template_highres, mode='full')
    lags = np.arange(-len(candidate_highres) + 1, len(template_highres))
    center = len(full_corr) // 2
    
    # Search within the allowed sub-sample shift window
    lag_window = int(max_shift * upsample)
    search_range = slice(center - lag_window, center + lag_window + 1)

    best_lag_index = np.argmax(full_corr[search_range])
    fractional_shift = lags[search_range][best_lag_index] / upsample

    # Apply the fractional shift to all channels of the candidate EI
    aligned_candidate = np.zeros_like(ei_candidate)
    for ch in range(C):
        interp_func = interp1d(t, ei_candidate[ch], kind='cubic', bounds_error=False, fill_value=0.0)
        shifted_time = t + fractional_shift
        aligned_candidate[ch] = interp_func(shifted_time)

    return aligned_candidate, fractional_shift


def compare_ei_subtraction(ei_a, ei_b, max_lag=3, p2p_thresh=30.0):
    """
    Compare two EIs by subtracting one from the other after sub-sample alignment,
    returning residual metrics and cosine similarity.
    """
    C, T = ei_a.shape
    assert ei_b.shape == (C, T), "EIs must have same shape"

    # Find the dominant channel on EI A to use as the alignment reference
    ref_chan = np.argmax(np.max(np.abs(ei_a), axis=1))
    
    # Align EI B to EI A
    aligned_b, fractional_shift = sub_sample_align_ei(ei_template=ei_a, ei_candidate=ei_b, ref_channel=ref_chan, upsample=10, max_shift=max_lag)

    p2p_a = np.ptp(ei_a, axis=1)
    good_channels = np.where(p2p_a > p2p_thresh)[0]

    per_channel_residuals = []
    per_channel_cosine_sim = []
    all_residuals = []

    for ch in good_channels:
        a = ei_a[ch]
        b = aligned_b[ch]
        
        # Mask: include samples on the channel only if they exceed 10% of that channel's peak-to-peak
        mask = np.abs(a) > 0.1 * np.max(np.abs(a))
        if not np.any(mask):
            continue
        
        a_masked, b_masked = a[mask], b[mask]
        residual = b_masked - a_masked
        per_channel_residuals.append(np.mean(residual))
        all_residuals.extend(residual)
        
        # Calculate local cosine similarity
        dot = np.dot(a_masked, b_masked)
        norm_product = np.linalg.norm(a_masked) * np.linalg.norm(b_masked) + 1e-8
        per_channel_cosine_sim.append(dot / norm_product)

    if not all_residuals:
        return {'mean_residual': np.nan, 'max_abs_residual': np.nan, 'good_channels': good_channels,
                'per_channel_residuals': per_channel_residuals, 'per_channel_cosine_sim': per_channel_cosine_sim,
                'fractional_shift': fractional_shift, 'p2p_a': p2p_a}
    
    mean_residual = np.mean(all_residuals)
    max_abs_residual = np.max(np.abs(all_residuals))

    return {
        'mean_residual': mean_residual,
        'max_abs_residual': max_abs_residual,
        'good_channels': good_channels,
        'per_channel_residuals': per_channel_residuals,
        'per_channel_cosine_sim': per_channel_cosine_sim,
        'fractional_shift': fractional_shift,
        'p2p_a': p2p_a
    }

2. Unit Clustering and Merging (clustering.py)
This file implements the clustering pipeline, including PCA for dimensionality reduction, HDBSCAN for initial clustering, and a customized routine to merge similar clusters based on both EI template similarity and separation in PC space.
# axolotl/clustering.py
"""
Functions for clustering spike waveforms, merging similar clusters,
and selecting the best cluster for further processing.
"""
import numpy as np
from typing import Union, Tuple
from itertools import combinations

from sklearn.decomposition import PCA
import networkx as nx
import hdbscan

# Local imports
from .comparison import compare_ei_subtraction, compare_eis
from .waveform_utils import check_2d_gap_peaks_valley


def cluster_separation_score(pcs, labels):
    """Calculates a separation score between pairs of clusters in PCA space.
    The score is the distance between cluster means normalized by the average spread (standard deviation)."""
    unique_labels = np.unique(labels)
    scores = []

    for A, B in combinations(unique_labels, 2):
        pcs_A = pcs[labels == A]
        pcs_B = pcs[labels == B]

        if len(pcs_A) == 0 or len(pcs_B) == 0:
            continue

        mu_A = pcs_A.mean(axis=0)
        mu_B = pcs_B.mean(axis=0)

        d_AB = np.linalg.norm(mu_A - mu_B)

        std_A = pcs_A.std(axis=0).mean()
        std_B = pcs_B.std(axis=0).mean()
        spread = (std_A + std_B) / 2

        # Separation score is distance normalized by average spread
        score = d_AB / (spread + 1e-8)
        scores.append({
            'pair': (A, B),
            'separation_score': score
        })

    return scores


def merge_similar_clusters_extra(
    snips,
    labels,
    max_lag=3,
    p2p_thresh=30.0,
    amp_thresh=-20,
    cos_thresh=0.8,
    pcs2=None,
    sep_thresh=3.0
):
    """
    Merge clusters whose EIs are highly similar, unless they are clearly
    separated in low-dimensional PC space (pcs2).
    
    Parameters:
        p2p_thresh (float): P2P threshold for determining 'good channels' for comparison.
        amp_thresh (float): Residual amplitude threshold for counting 'bad channels'.
        cos_thresh (float): Cosine similarity threshold for the most permissive merge case.
        sep_thresh (float): PC-space separation score threshold. Scores below this are allowed to merge.
    """
    cluster_ids = sorted(np.unique(labels))
    cluster_spike_idx = {k: np.where(labels == k)[0] for k in cluster_ids}
    n_clusters = len(cluster_ids)
    id2idx = {cid: i for i, cid in enumerate(cluster_ids)}

    cluster_eis = []
    cluster_vars = []
    for k in cluster_ids:
        inds = cluster_spike_idx[k]
        ei_k = snips[:, :, inds].mean(axis=2)
        cluster_eis.append(ei_k)
        
        # Calculate local variance (a basic SNR proxy)
        peak_idx = np.argmin(ei_k, axis=1)
        var_ch = np.array([
            np.var(snips[ch, max(0, i-1):i+2, inds]) if 1 <= i < ei_k.shape[1]-1 else 0.0
            for ch, i in enumerate(peak_idx)
        ])
        cluster_vars.append(var_ch)

    sim = np.eye(n_clusters)
    n_bad_ch = np.zeros((n_clusters, n_clusters), dtype=int)

    # 1. Compute Similarity (EI Template Matching) and Bad Channel Count
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            ei_a = cluster_eis[i]
            ei_b = cluster_eis[j]
            var_a = cluster_vars[i]
            
            # Compare EIs using subtraction method
            res_ab = compare_ei_subtraction(ei_a, ei_b, max_lag=max_lag, p2p_thresh=p2p_thresh)
            
            if not res_ab['good_channels'].size > 0:
                sim[i,j] = sim[j,i] = 0.0
                continue
                
            res = np.array(res_ab['per_channel_residuals'])
            p2p_a = res_ab['p2p_a']
            good_channels = res_ab['good_channels']
            cos_sim = np.array(res_ab['per_channel_cosine_sim'])
            
            ch_weights = p2p_a[good_channels]
            
            # Use a simple SNR mask for weighting
            snr_score = 1 / (1 + var_a[good_channels] / (p2p_a[good_channels]**2 + 1e-3))
            snr_mask = snr_score > 0.5
            
            if snr_mask.sum() == 0:
                weighted_cos_sim = 0.0
                neg_inds_count = 0
            else:
                res_subset = res[snr_mask]
                cos_sim_masked = cos_sim[snr_mask]
                ch_weights_masked = ch_weights[snr_mask]
                
                # Weighted cosine similarity score
                weighted_cos_sim = np.average(cos_sim_masked, weights=ch_weights_masked)
                
                # Count 'bad channels': residuals below the negative amplitude threshold
                neg_inds_count = np.sum(res_subset < amp_thresh)

            sim[i, j] = sim[j, i] = weighted_cos_sim
            n_bad_ch[i, j] = n_bad_ch[j, i] = neg_inds_count

    # 2. Compute Separation Score (PC Space)
    sep = None
    if pcs2 is not None:
        scores = cluster_separation_score(pcs2, labels)
        sep = np.zeros((n_clusters, n_clusters))
        for score_dict in scores:
            id_a, id_b = score_dict['pair']
            idx_a, idx_b = id2idx[id_a], id2idx[id_b]
            sep[idx_a, idx_b] = sep[idx_b, idx_a] = score_dict['separation_score']

    # 3. Perform Greedy Merging
    cluster_sizes = {cid: len(cluster_spike_idx[cid]) for cid in cluster_ids}
    sorted_ids = sorted(cluster_ids, key=lambda c: cluster_sizes[c], reverse=True)
    assigned = set()
    merged_clusters = []

    for cid in sorted_ids:
        if cid in assigned: continue
        group = [cid]
        assigned.add(cid)
        changed = True
        while changed:
            changed = False
            for other in sorted_ids:
                if other in assigned: continue
                accept = False
                for existing in group:
                    i, j = id2idx[existing], id2idx[other]
                    
                    # Merge criteria based on Similarity (sim_ok) and Separation (sep_ok)
                    sim_ok = (
                        (sim[i, j] >= 0.95 and n_bad_ch[i, j] <= 6) or
                        (sim[i, j] >= 0.90 and n_bad_ch[i, j] <= 4) or
                        (sim[i, j] >= 0.80 and n_bad_ch[i, j] == 2) or
                        (sim[i, j] >= cos_thresh and n_bad_ch[i, j] == 0)
                    )
                    sep_ok = (sep is None) or (sep[i, j] <= sep_thresh)
                    
                    if sim_ok and sep_ok:
                        accept = True
                        break
                if accept:
                    group.append(other)
                    assigned.add(other)
                    changed = True
        merged_spikes = np.concatenate([cluster_spike_idx[c] for c in group])
        merged_clusters.append(np.sort(merged_spikes))

    return merged_clusters, sim, n_bad_ch


def cluster_spike_waveforms(
    snips: np.ndarray,
    ei: np.ndarray,
    p2p_threshold: float = 15,
    min_chan: int = 30,
    max_chan: int = 80,
    hdbscan_min_cluster_size: int = 75,
    hdbscan_min_samples: int = 10,
    merge_p2p_thresh: float = 30.0,
    merge_cos_thresh: float = 0.8,
    merge_amp_thresh: float = -20,
    merge_sep_thresh: float = 3.0,
    return_debug: bool = False
) -> Union[list[dict], Tuple]:
    """
    Cluster spike waveforms based on selected EI channels and merge using EI similarity.
    Uses HDBSCAN for density-based clustering.
    
    Parameters are extended to allow tuning of HDBSCAN and merging logic from config.
    """
    # 1. Select channels for dimensionality reduction
    ei_p2p = np.ptp(ei, axis=1)
    selected_channels = np.where(ei_p2p > p2p_threshold)[0]
    
    if len(selected_channels) > max_chan:
        selected_channels = np.argsort(ei_p2p)[-max_chan:]
    elif len(selected_channels) < min_chan:
        selected_channels = np.argsort(ei_p2p)[-min_chan:]
    
    if len(selected_channels) == 0:
        selected_channels = np.argsort(ei_p2p)[-min_chan:]

    selected_channels = np.sort(selected_channels)
    
    snips_sel = snips[selected_channels, :, :]
    N = snips_sel.shape[2]
    snips_flat = snips_sel.transpose(2, 0, 1).reshape(N, -1)

    # 2. Project into 2D PC space
    pcs = PCA(n_components=2).fit_transform(snips_flat)
    
    # 3. HDBSCAN Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, 
                              min_samples=hdbscan_min_samples,
                              prediction_data=True)
    labels = clusterer.fit_predict(pcs)

    # 4. Merge similar clusters based on EI and PC separation
    merged_clusters, sim, n_bad_channels = merge_similar_clusters_extra(
        snips=snips, labels=labels, max_lag=3, 
        p2p_thresh=merge_p2p_thresh,
        amp_thresh=merge_amp_thresh, 
        cos_thresh=merge_cos_thresh, 
        pcs2=pcs[:, :2], 
        sep_thresh=merge_sep_thresh
    )

    # 5. Format output
    output = []
    for inds in merged_clusters:
        if inds.size > 0:
            ei_cluster = np.mean(snips[:, :, inds], axis=2)
            output.append({'inds': inds, 'ei': ei_cluster, 'channels': selected_channels})

    if return_debug:
        # Prepare debug output for visualization
        cluster_spike_indices = {k: np.where(labels == k)[0] for k in np.unique(labels)}
        cluster_eis = [np.mean(snips[:, :, inds], axis=2) for inds in cluster_spike_indices.values() if inds.size > 0]
        
        # Map original HDBSCAN cluster ID to the final merged group ID
        cluster_to_merged_group = {}
        for orig_id, orig_inds in cluster_spike_indices.items():
            for g, merged_inds in enumerate(merged_clusters):
                # Check if the original cluster indices are a subset of the final merged indices
                if np.all(np.in1d(orig_inds, merged_inds)):
                    cluster_to_merged_group[orig_id] = g
                    break
        
        return output, pcs, labels, sim, n_bad_channels, cluster_eis, cluster_to_merged_group
    else:
        return output


def select_cluster_with_largest_waveform(
    clusters: list[dict],
    ref_channel: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Select the cluster with the largest (most negative) EI amplitude on the reference channel.
    """
    if not clusters:
        raise ValueError("Input cluster list is empty.")
        
    amplitudes = [np.min(cl['ei'][ref_channel, :]) for cl in clusters]
    best_idx = int(np.argmin(amplitudes))
    best = clusters[best_idx]
    return best['ei'], best['inds'], best['channels'], best_idx


def select_cluster_by_ei_similarity_ram(
    snips: np.ndarray,
    clusters: list[dict],
    reference_ei: np.ndarray,
    similarity_threshold: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Merge clusters based on EI similarity and select the one most similar to a reference EI.
    """
    if not clusters:
        raise ValueError("Input cluster list is empty.")

    cluster_eis = [cl['ei'] for cl in clusters]
    sim = compare_eis(cluster_eis)

    G = nx.from_numpy_array(sim >= similarity_threshold)
    merged_groups = list(nx.connected_components(G))
    
    merged_clusters = [np.sort(np.concatenate([clusters[i]['inds'] for i in group])) for group in merged_groups]
    merged_eis = [np.mean(snips[:, :, inds], axis=2) for inds in merged_clusters if inds.size > 0] 

    if not merged_eis:
         raise ValueError("No merged clusters with spikes found.")

    similarities = compare_eis(merged_eis, ei_template=reference_ei).flatten()
    best_idx = int(np.argmax(similarities))
    
    final_inds = merged_clusters[best_idx]
    final_ei = merged_eis[best_idx]
    
    original_cluster_idx = list(merged_groups[best_idx])[0]
    final_channels = clusters[original_cluster_idx]['channels']

    return final_ei, final_inds, final_channels, best_idx

3. Template Subtraction (Peeling) (subtraction.py)
This file contains the logic for peeling—the process of subtracting the unit's template from the raw data trace so the remaining residual signal can be processed for smaller units. It includes both high-level PCA-clustered subtraction and a Numba-accelerated helper for efficient, in-place data modification.
# axolotl/subtraction.py
"""
Functions for subtracting found neuron templates from the raw data trace (peeling).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numba import njit

def subtract_pca_cluster_means_ram(snippets, baselines, spike_times, segment_len=100_000, n_clusters=5, offset_window=(-5,10)):
    """
    Subtracts PCA-clustered mean waveforms from baseline-corrected spike snippets for a single channel.
    This method attempts to model and subtract common noise and template variations on a channel.

    Parameters
    ----------
    snippets : np.ndarray, shape (n_spikes, snip_len)
        Array for a single channel.
    baselines : np.ndarray, shape (n_segments,)
        Array of mean baseline per segment for this channel.
    spike_times : np.ndarray, shape (n_spikes,)
        Array of spike times (in samples).
    segment_len : int
        Segment size used for baseline estimation.
    n_clusters : int
        Number of PCA/k-means clusters to use.
    offset_window: tuple
        Window relative to the peak for PCA and subtraction.

    Returns
    -------
    residuals : np.ndarray, shape (n_spikes, snip_len)
        int16 array of subtracted residuals with baseline added back.
    scale_factors : np.ndarray, shape (n_spikes,)
        float32 array of amplitude scaling per spike.
    cluster_ids : np.ndarray, shape (n_spikes,)
        int32 array of cluster IDs.
    """
    # --- Baseline subtraction ---
    segment_ids = spike_times // segment_len
    segment_ids = np.clip(segment_ids, 0, len(baselines) - 1)
    baseline_per_spike = baselines[segment_ids][:, np.newaxis]
    snippets_bs = snippets - baseline_per_spike

    # --- Template and window definition ---
    template = np.mean(snippets_bs, axis=0)
    neg_peak_idx = np.argmin(template)
    w_start = max(0, neg_peak_idx + offset_window[0])
    w_end = min(snippets.shape[1], neg_peak_idx + offset_window[1])
    window = slice(w_start, w_end)

    # --- PCA and clustering (for modeling variations) ---
    pca = PCA(n_components=5)
    # Use only the window for PCA
    reduced = pca.fit_transform(snippets_bs[:, window])
    cluster_ids = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit_predict(reduced)

    # --- Subtract cluster means (only in window) ---
    residuals = np.copy(snippets_bs)
    scale_factors = np.ones(snippets_bs.shape[0], dtype=np.float32) # Placeholder
    
    for c in range(n_clusters):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) == 0:
            continue
        cluster_mean = np.mean(snippets_bs[idx, window], axis=0)
        residuals[idx, window] -= cluster_mean

    # --- Restore baseline, clip, convert ---
    residuals += baseline_per_spike
    residuals = np.clip(residuals, -32768, 32767).astype(np.int16)

    return residuals, scale_factors, cluster_ids


@njit(cache=True)
def _apply_residuals_channel(raw_data_ch, residuals, write_locs):
    """Numba-accelerated helper to apply residuals for a SINGLE channel."""
    total_samples = len(raw_data_ch)
    n_spikes = len(write_locs)
    snip_len = residuals.shape[1]
    
    for i in range(n_spikes):
        loc = write_locs[i]
        end = loc + snip_len
        # Ensure snippet window is within the bounds of the recording
        if loc >= 0 and end <= total_samples:
            raw_data_ch[loc:end] = residuals[i, :]

def apply_residuals(
    raw_data: np.ndarray,
    residual_snips_per_channel: dict,
    write_locs: np.ndarray,
    selected_channels: np.ndarray,
    total_samples: int,
    is_ram: bool = True
):
    """
    Applies residual snippets to time-major data by calling a fast Numba kernel.
    This modifies the raw_data array in-place.
    """
    if not is_ram:
        raise NotImplementedError("Disk-based modification is not supported.")

    # This loop runs in normal Python, iterating over channels
    for ch in selected_channels:
        if ch not in residual_snips_per_channel:
            continue
            
        residuals = residual_snips_per_channel[ch]

        if residuals.shape[0] != len(write_locs):
            raise ValueError(f"Mismatch between residuals and write_locs for channel {ch}")

        # Call the fast Numba kernel for the actual data modification
        _apply_residuals_channel(raw_data[:, ch], residuals, write_locs)


def subtract_scaled_template_ram(snippets, template):
    """
    Subtracts a scaled version of a template from a set of snippets.
    This is a simplified, robust subtraction function, often used as a fallback.

    Parameters
    ----------
    snippets : np.ndarray, shape (n_spikes, snip_len)
        Array for a single channel.
    template : np.ndarray, shape (snip_len,)
        Array of the template to subtract.

    Returns
    -------
    residuals : np.ndarray, shape (n_spikes, snip_len)
        int16 array of subtracted residuals.
    """
    n_spikes, snip_len = snippets.shape
    
    template_f32 = template.astype(np.float32)
    snippets_f32 = snippets.astype(np.float32)

    # Calculate optimal scaling factor via linear regression (dot product method)
    dot_product = np.dot(snippets_f32, template_f32)
    template_norm_sq = np.dot(template_f32, template_f32) + 1e-6
    
    scale_factors = dot_product / template_norm_sq
    scale_factors = np.clip(scale_factors, 0.5, 2.0) # Clip scale factor to typical range
    
    residuals_f32 = snippets_f32 - scale_factors[:, np.newaxis] * template_f32
    
    residuals = np.clip(residuals_f32, -32768, 32767).astype(np.int16)
    
    return residuals

