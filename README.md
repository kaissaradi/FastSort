# FastSort
A fast gpu spike sorter

FastSort: A GPU-Accelerated Spike Sorter
FastSort is a high-performance, batch-processing spike sorting pipeline designed for speed and accuracy on dense electrophysiology recordings. It leverages NVIDIA GPUs via the RAPIDS ecosystem (CuPy and cuML) to perform all computationally intensive tasks, from spike detection to clustering.
The core pipeline is built on a modern, non-iterative workflow:
 * Global Detection: All spikes are detected across all channels simultaneously.
 * Deduplication: A robust algorithm ensures each unique spike event is represented only once.
 * PCA & HDBSCAN: Spikes are clustered in a low-dimensional space using the powerful HDBSCAN algorithm, which can identify noise and does not require a pre-set number of clusters.
 * Phy Output: Results are saved in the industry-standard, Phy-compatible format.
Installation
Prerequisites
 * An NVIDIA GPU with CUDA support.
 * The NVIDIA CUDA Toolkit installed (version 11.x is recommended).
 * Python 3.8+
Steps
 * Clone the repository:
   git clone <your-repository-url>
cd FastSort

 * Install the required Python packages using pip. This will automatically download the correct GPU libraries based on your CUDA version.
   pip install -r requirements.txt

How to Run
 * Configure your sort: Open the config.yaml file and edit the parameters to match your recording and desired settings.
   * raw_data_path: Path to your .bin or .dat raw data file.
   * channel_map_path: Path to your .npy channel positions file.
   * output_dir: Directory where the results will be saved.
   * refractory_period_ms: The time window (in ms) for removing duplicate spikes. 0.5 is a good default.
   * n_channels_for_pca: The number of neighboring channels to use when creating features for clustering.
   * n_pca_components: The number of dimensions to reduce to before clustering.
   * min_cluster_size: The minimum number of spikes required to form a cluster in HDBSCAN.
 * Execute the pipeline: Run the main script from your terminal, pointing it to your configuration file.
   python run_sort.py --config config.yaml

 * View the results: Once finished, the output_dir will contain all the necessary files to be viewed in Phy.
Development Log & Methodology
This project was created by refactoring the original "Axolotl" spike sorter into a fully modern, GPU-accelerated pipeline. The following key changes were implemented:
 * Architectural Shift: The core logic was transformed from a slow, iterative "peeling" (detect-subtract-repeat) workflow into a single-pass, batch-processing system. This is the primary source of the massive speed improvement.
 * GPU Acceleration: All computationally heavy steps were offloaded to the GPU:
   * NumPy was replaced with CuPy for all array operations on the raw data.
   * Scikit-learn and hdbscan were replaced with their RAPIDS cuML equivalents for PCA and HDBSCAN, respectively.
 * Clustering Algorithm Upgrade: The original KMeans clustering was replaced with HDBSCAN. This is a significant improvement, as HDBSCAN can automatically determine the number of clusters and explicitly identify noise spikes, leading to cleaner and more reliable results.
 * New Modules & File Cleanup:
   * deduplication.py: A new, critical module was created to handle the logic of finding unique spike events from raw, multi-channel detections.
   * run_sort.py: A new main script was created to orchestrate the new batch-processing pipeline.
   * Obsolete Files Removed: subtraction.py and collision.py, which were central to the old peeling method, were removed to create a clean, focused codebase.
Testing and Validation Plan
To ensure the sorter is working correctly and producing high-quality results, the following tests should be performed.
1. Data Integrity Checks
 * [ ] File Creation: Verify that after a successful run, the output_dir contains all expected Phy files: spike_times.npy, spike_clusters.npy, templates.npy, channel_positions.npy, channel_map.npy, params.yml, and cluster_group.tsv.
 * [ ] Data Shape and Type: Load each .npy file and confirm that the array shapes and data types match the Phy specification (e.g., spike_times is int64, templates is float32, etc.).
 * [ ] Spike Count: Ensure that the number of spikes in spike_times.npy matches the number of entries in spike_clusters.npy.
2. Visual & Qualitative Validation (in Phy)
 * [ ] Waveform Sanity Check: Open the results in Phy and inspect the templates for each cluster. Do they look like neural waveforms?
 * [ ] Cluster Separation: Use the feature view in Phy to check if clusters appear reasonably well-separated.
 * [ ] Autocorrelograms: Check the autocorrelograms for each cluster. "Good" units should show a clear refractory period (a dip at t=0). The presence of many spikes within the refractory period might indicate a multi-unit cluster.
3. Performance Benchmarking
 * [ ] Runtime Measurement: Time a full run on a standard dataset. Record the duration, dataset size (in GB), and GPU model used.
 * [ ] Scalability Test: Run the sorter on datasets of increasing length (e.g., 5 min, 30 min, 2 hours) to ensure performance scales linearly.
4. Ground Truth Validation (Optional but highly recommended)
 * [ ] Run on Simulated Data: Use a tool like MEArec to generate a simulated dataset where the ground truth spike times and unit IDs are known.
 * [ ] Calculate Metrics: Run the sorter on the simulated data and compare its output to the ground truth. Calculate key performance metrics such as accuracy, precision, and recall for spike detection and clustering. This is the gold standard for validating a spike sorter's performance.
