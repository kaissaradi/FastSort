# run_axolotl_gpu.py
"""
Main execution script for the Axolotl-GPU spike sorting pipeline.

This script orchestrates a high-speed, batch-processing workflow that leverages
NVIDIA GPUs for all heavy computational tasks, including detection, feature
extraction, and clustering.
"""
import yaml
import argparse
import numpy as np
import cupy as cp
import os
import time

# --- GPU-accelerated libraries ---
from cuml.decomposition import PCA as cuPCA
from cuml.cluster import HDBSCAN as cuHDBSCAN

# --- Axolotl modules ---
from axolotl.io import (
    load_raw_binary,
    load_channel_map,
    save_phy_results,
    save_cluster_group_tsv
)
from axolotl.detection import detect_spikes_globally_gpu
from axolotl.deduplication import deduplicate_spikes_gpu
from axolotl.clustering import get_cluster_templates_gpu

def main(config_path):
    """
    Loads configuration, data, and runs the GPU-accelerated spike sorting pipeline.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Configuration Parameters ---
    # Paths
    raw_data_path = config['paths']['raw_data_path']
    channel_map_path = config['paths']['channel_map_path']
    output_dir = config['paths']['output_dir']

    # Recording
    n_channels = config['recording']['n_channels']
    sampling_rate = config['recording']['sampling_rate']
    dtype = config['recording']['dtype']

    # Pipeline settings
    window_pre = config['pipeline']['window_pre_samples']
    window_post = config['pipeline']['window_post_samples']
    refractory_ms = config['pipeline'].get('refractory_period_ms', 0.5)
    refractory_samples = int(refractory_ms / 1000 * sampling_rate)

    # Snippet extraction settings
    n_channels_pca = config['clustering'].get('n_channels_for_pca', 10)
    
    # Clustering settings
    n_pca_components = config['clustering'].get('n_pca_components', 10)
    min_cluster_size = config['clustering'].get('min_cluster_size', 25)

    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load Data and Transfer to GPU ---
    start_time_total = time.time()
    raw_data_cpu = load_raw_binary(raw_data_path, n_channels, dtype)
    channel_map_cpu = load_channel_map(channel_map_path)

    print("\n--- Moving data to GPU ---")
    raw_data_gpu = cp.asarray(raw_data_cpu, dtype=cp.float32)
    channel_map_gpu = cp.asarray(channel_map_cpu)
    
    # --- 2. Preprocessing: Baseline Correction (GPU) ---
    print("Performing baseline correction on GPU...")
    # Simple median subtraction is fast and robust on the GPU
    raw_data_gpu -= cp.median(raw_data_gpu, axis=0, keepdims=True)

    # --- 3. Global Spike Detection (GPU) ---
    print("\n--- Detecting Spikes ---")
    # Using a robust threshold: 5 * Median Absolute Deviation
    noise_mad_gpu = cp.median(cp.abs(raw_data_gpu), axis=0)
    thresholds_gpu = -5 * noise_mad_gpu / 0.6745
    
    raw_times, raw_channels = detect_spikes_globally_gpu(
        raw_data_gpu, sampling_rate, thresholds_gpu
    )

    # --- 4. Deduplication (GPU) ---
    print("\n--- Deduplicating Spikes ---")
    unique_times, unique_channels = deduplicate_spikes_gpu(
        raw_times, raw_channels, raw_data_gpu, refractory_samples=refractory_samples
    )
    
    # --- 5. Snippet Extraction for PCA (GPU) ---
    print("\n--- Extracting Snippets for PCA ---")
    n_spikes = len(unique_times)
    win_len = window_post - window_pre
    
    # Find the `n_channels_pca` nearest channels for each spike
    dist_sq = cp.sum((channel_map_gpu[unique_channels, None, :] - channel_map_gpu[None, :, :])**2, axis=2)
    pca_channel_indices = cp.argsort(dist_sq, axis=1)[:, :n_channels_pca]
    
    # Prepare indices for advanced GPU slicing
    time_indices = unique_times[:, None] + cp.arange(window_pre, window_post, dtype=cp.int32)
    
    # Extract snippets using advanced indexing (gather operation)
    snippets_gpu = raw_data_gpu[time_indices[:, :, None], pca_channel_indices[:, None, :]]
    # Reshape to (n_channels_pca, n_samples, n_spikes)
    snippets_for_pca = snippets_gpu.transpose(2, 1, 0)
    
    # --- 6. PCA and HDBSCAN Clustering (GPU) ---
    print("\n--- Running PCA and Clustering ---")
    features_gpu = snippets_for_pca.transpose(2, 0, 1).reshape(n_spikes, -1)
    
    pca_gpu = cuPCA(n_components=n_pca_components, svd_solver='jacobi')
    pcs_gpu = pca_gpu.fit_transform(features_gpu)
    
    clusterer_gpu = cuHDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    labels_gpu = clusterer_gpu.fit_predict(pcs_gpu)
    
    n_clusters_found = int(cp.max(labels_gpu)) + 1
    n_noise = int(cp.sum(labels_gpu == -1))
    print(f"HDBSCAN found {n_clusters_found} clusters and {n_noise:,} noise points.")
    
    # --- 7. Template Generation (GPU) ---
    print("\n--- Generating Templates ---")
    # Extract snippets on ALL channels for final templates
    all_chan_snippets_gpu = raw_data_gpu[time_indices[:, :, None], cp.arange(n_channels)[None, None, :]]
    all_chan_snippets_gpu = all_chan_snippets_gpu.transpose(2, 1, 0)
    
    templates_gpu = get_cluster_templates_gpu(all_chan_snippets_gpu, labels_gpu)

    # --- 8. Transfer Final Results to CPU ---
    print("\n--- Transferring results to CPU for saving ---")
    # Filter out noise spikes for saving
    non_noise_mask = (labels_gpu != -1)
    
    final_spike_times = cp.asnumpy(unique_times[non_noise_mask])
    final_spike_clusters = cp.asnumpy(labels_gpu[non_noise_mask])
    final_templates = cp.asnumpy(templates_gpu)
    unique_cluster_ids = np.unique(final_spike_clusters)

    # --- 9. Save to Phy Format ---
    save_phy_results(
        output_dir=output_dir,
        spike_times=final_spike_times,
        spike_clusters=final_spike_clusters,
        templates=final_templates,
        channel_map=channel_map_cpu,
        config=config
    )
    
    save_cluster_group_tsv(
        output_dir=output_dir,
        cluster_ids=unique_cluster_ids
    )

    end_time_total = time.time()
    print(f"\n✅ Axolotl-GPU pipeline finished in {end_time_total - start_time_total:.2f} seconds.")
    print(f"Found and saved {len(unique_cluster_ids)} clusters.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Axolotl-GPU spike sorting pipeline.")
    parser.add_argument('--config', required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    main(args.config)

