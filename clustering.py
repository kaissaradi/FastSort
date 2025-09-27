# axolotl/clustering.py
"""
Functions for clustering spike waveforms, generating templates, and classifying
cluster quality. Includes new GPU-native functions and legacy CPU methods.
"""
import numpy as np
import cupy as cp
from typing import Union, Tuple
from itertools import combinations

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import networkx as nx

from .comparison import compare_ei_subtraction, compare_eis
from .waveform_utils import check_2d_gap_peaks_valley

# --------------------------------------------------------------------------
# --- New GPU-Accelerated & Classification Functions ---
# --------------------------------------------------------------------------

def get_cluster_templates_gpu(
    snippets_gpu: cp.ndarray,
    labels_gpu: cp.ndarray
) -> cp.ndarray:
    """
    Computes the mean waveform (template) for each cluster entirely on the GPU.

    Parameters
    ----------
    snippets_gpu : cp.ndarray, shape (C, T, N)
        The spike snippets for all detected spikes, resident on the GPU.
    labels_gpu : cp.ndarray, shape (N,)
        The cluster ID assigned to each spike.

    Returns
    -------
    cp.ndarray, shape (n_clusters, n_samples, n_channels)
        The mean templates for each cluster, in Phy-compatible format.
    """
    if labels_gpu.size == 0:
        return cp.empty((0, snippets_gpu.shape[1], snippets_gpu.shape[0]), dtype=cp.float32)

    unique_labels = cp.unique(labels_gpu)
    # Exclude noise cluster (label -1) if present
    unique_labels = unique_labels[unique_labels != -1]
    
    if unique_labels.size == 0:
        return cp.empty((0, snippets_gpu.shape[1], snippets_gpu.shape[0]), dtype=cp.float32)

    n_clusters = int(cp.max(unique_labels)) + 1
    n_channels, n_samples, _ = snippets_gpu.shape
    templates_gpu = cp.zeros((n_clusters, n_channels, n_samples), dtype=cp.float32)

    for k in unique_labels:
        k = int(k)
        # Use CuPy boolean indexing to select snippets for the current cluster
        cluster_snippets = snippets_gpu[:, :, labels_gpu == k]
        if cluster_snippets.shape[2] > 0:
            templates_gpu[k, :, :] = cp.mean(cluster_snippets, axis=2)
    
    # Phy expects templates in (n_clusters, n_samples, n_channels), so we transpose
    return cp.transpose(templates_gpu, (0, 2, 1))


def classify_clusters(
    templates: np.ndarray,
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    sampling_rate: int
) -> dict:
    """
    Classifies clusters as 'good', 'mua', or 'noise' based on simple metrics.
    This function runs on the CPU after results are gathered from the GPU.

    Parameters
    ----------
    templates : np.ndarray, shape (n_units, n_samples, n_channels)
        The mean waveform (EI) for each unit.
    spike_times : np.ndarray, shape (n_spikes,)
        The sample index of each detected spike.
    spike_clusters : np.ndarray, shape (n_spikes,)
        The cluster ID assigned to each spike.
    sampling_rate : int
        The sampling rate of the recording in Hz.

    Returns
    -------
    dict
        A dictionary mapping each cluster_id to a quality label ('good', 'mua', 'noise').
    """
    cluster_groups = {}
    unique_clusters = np.unique(spike_clusters)

    for cluster_id in unique_clusters:
        # --- Metric 1: Refractory Period Violations ---
        cluster_spike_times = spike_times[spike_clusters == cluster_id]
        
        # Calculate inter-spike intervals (ISIs) in milliseconds
        isis_ms = np.diff(cluster_spike_times) / sampling_rate * 1000
        
        # Count violations (e.g., ISIs less than 1.5 ms)
        refractory_violations = np.sum(isis_ms < 1.5)
        violation_rate = refractory_violations / len(cluster_spike_times) if len(cluster_spike_times) > 0 else 0

        # --- Metric 2: Waveform Amplitude ---
        # Find the peak-to-peak amplitude on the channel with the largest signal
        template = templates[cluster_id]
        p2p_per_channel = np.ptp(template, axis=0)
        amplitude = np.max(p2p_per_channel)

        # --- Classification Logic (example thresholds) ---
        if violation_rate < 0.1 and amplitude > 75:
            group = 'good'
        elif violation_rate < 0.25 and amplitude > 40:
            group = 'mua'  # Multi-Unit Activity
        else:
            group = 'noise'
        
        cluster_groups[cluster_id] = group
    
    return cluster_groups


# --------------------------------------------------------------------------
# --- Legacy CPU-Based Functions (Kept for Reference) ---
# --------------------------------------------------------------------------

def cluster_separation_score(pcs, labels):
    """Calculates a separation score between pairs of clusters in PCA space."""
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
        peak_idx = np.argmin(ei_k, axis=1)
        var_ch = np.array([
            np.var(snips[ch, max(0, i-1):i+2, inds]) if 1 <= i < ei_k.shape[1]-1 else 0.0
            for ch, i in enumerate(peak_idx)
        ])
        cluster_vars.append(var_ch)

    sim = np.eye(n_clusters)
    n_bad_ch = np.zeros((n_clusters, n_clusters), dtype=int)

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            ei_a = cluster_eis[i]
            ei_b = cluster_eis[j]
            var_a = cluster_vars[i]
            res_ab = compare_ei_subtraction(ei_a, ei_b, max_lag=max_lag, p2p_thresh=p2p_thresh)
            
            if not res_ab['good_channels'].size > 0:
                sim[i,j] = sim[j,i] = 0.0
                continue
                
            res = np.array(res_ab['per_channel_residuals'])
            p2p_a = res_ab['p2p_a']
            good_channels = res_ab['good_channels']
            cos_sim = np.array(res_ab['per_channel_cosine_sim'])
            
            ch_weights = p2p_a[good_channels]
            snr_score = 1 / (1 + var_a[good_channels] / (p2p_a[good_channels]**2 + 1e-3))
            snr_mask = snr_score > 0.5
            
            if snr_mask.sum() == 0:
                weighted_cos_sim = 0.0
                neg_inds_count = 0
            else:
                res_subset = res[snr_mask]
                cos_sim_masked = cos_sim[snr_mask]
                ch_weights_masked = ch_weights[snr_mask]
                weighted_cos_sim = np.average(cos_sim_masked, weights=ch_weights_masked)
                neg_inds_count = np.sum(res_subset < amp_thresh)

            sim[i, j] = sim[j, i] = weighted_cos_sim
            n_bad_ch[i, j] = n_bad_ch[j, i] = neg_inds_count

    sep = None
    if pcs2 is not None:
        scores = cluster_separation_score(pcs2, labels)
        sep = np.zeros((n_clusters, n_clusters))
        for score_dict in scores:
            id_a, id_b = score_dict['pair']
            idx_a, idx_b = id2idx[id_a], id2idx[id_b]
            sep[idx_a, idx_b] = sep[idx_b, idx_a] = score_dict['separation_score']

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
    k_start: int = 3,
    p2p_threshold: float = 15,
    min_chan: int = 30,
    max_chan: int = 80,
    return_debug: bool = False
) -> Union[list[dict], Tuple]:
    """
    Cluster spike waveforms based on selected EI channels and merge using EI similarity.
    """
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

    pcs = PCA(n_components=2).fit_transform(snips_flat)
    kmeans = KMeans(n_clusters=k_start, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pcs)

    if len(np.unique(labels)) < 8:
        labels_updated = labels.copy()
        next_label = labels_updated.max() + 1
        to_check = list(np.unique(labels_updated))
        while to_check:
            cl = to_check.pop(0)
            mask = labels_updated == cl
            if np.sum(mask) < 40: continue
            pc_vals = pcs[mask, :2]
            split_result = check_2d_gap_peaks_valley(pc_vals, angle_step=10, min_valley_frac=0.25)
            if split_result:
                g1_mask, g2_mask = split_result
                if np.sum(g1_mask) < 20 or np.sum(g2_mask) < 20: continue
                cluster_indices = np.where(mask)[0]
                labels_updated[cluster_indices[g2_mask]] = next_label
                to_check.extend([cl, next_label])
                next_label += 1
        labels = labels_updated

    merged_clusters, sim, n_bad_channels = merge_similar_clusters_extra(
        snips, labels, max_lag=3, p2p_thresh=30.0,
        amp_thresh=-20, cos_thresh=0.75, pcs2=pcs[:, :2], sep_thresh=8.0
    )

    output = []
    for inds in merged_clusters:
        if inds.size > 0:
            ei_cluster = np.mean(snips[:, :, inds], axis=2)
            output.append({'inds': inds, 'ei': ei_cluster, 'channels': selected_channels})

    if return_debug:
        cluster_spike_indices = {k: np.where(labels == k)[0] for k in np.unique(labels)}
        cluster_eis = [np.mean(snips[:, :, inds], axis=2) for inds in cluster_spike_indices.values() if inds.size > 0]
        cluster_to_merged_group = {}
        for orig_id, orig_inds in cluster_spike_indices.items():
            for g, merged_inds in enumerate(merged_clusters):
                if set(orig_inds).issubset(merged_inds):
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
