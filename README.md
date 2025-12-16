Few-Shot Anomaly Detection in the Textile Industry: A Pragmatic Framework

📄 Overview
This repository contains the source code for the paper:"From Benchmarks to the Factory Floor: A Pragmatic Framework for Few-Shot Anomaly Detection in the Textile Industry"
The project presents a rigorous comparative analysis of unsupervised anomaly detection methods (Classical: Single-Centroid, KMeans, OCSVM; Deep Learning: VAE, AnoGAN) in an extreme few-shot setting (10 normal samples for training). The framework is optimized for industrial viability, emphasizing speed, stability, and interpretability over marginal accuracy gains.
🚀 Key Features

Few-Shot Protocol: Standardized 10-shot training for fair comparison across all methods and datasets.
Comprehensive Benchmarking: Evaluated across five diverse industrial and benchmark datasets.
Automated Feature Extraction: Utilizes a pre-trained EfficientNet-B7 model (via Hugging Face Transformers) for high-quality feature vectors.
Statistically Rigorous: Automated report generation includes F1-scores, AUROC, AUPR, training/inference times, and statistical analysis (Wilcoxon Signed-Rank Test).
Reproducibility: Handles data loading, feature caching, hyperparameter tuning emulation, and generates qualitative showcase reports.

🛠️ Setup and Installation
1. Environment Setup
It is highly recommended to use a dedicated Python virtual environment (e.g., using conda or venv).
# Example using conda
conda create -n anomaly-env python=3.9
conda activate anomaly-env

2. Dependencies
Install the necessary libraries. A GPU with CUDA support is required for optimal performance due to the heavy use of deep learning.
# Install core dependencies (replace cu121 with your CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other libraries
pip install numpy pandas scikit-learn transformers tqdm matplotlib seaborn pillow openpyxl xlsxwriter

3. Data Preparation
The project requires a specific directory structure under the data/ folder, set by CONFIG["DATASET_ROOT"] = "data". Download the datasets and organize them as follows:



Dataset Name (Folder in data/)
Source Link
Notes



dataset_kaggle
Kaggle: AITEX Fabric Image Database
Use the provided image files.
https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database

dataset_amazon
GitHub: Amazon Spot Diff
Use image files from the official repo.
https://github.com/amazon-science/spot-diff

dataset_mvtec
MVTec AD
Standard MVTec AD dataset.
https://www.mvtec.com/company/research/datasets/mvtec-ad

dataset_mvtec2
MVTec 2D LOCO
Standard MVTec 2D LOCO dataset.
https://www.mvtec.com/company/research/datasets/mvtec-loco

dataset_proprietary
Not publicly available
Contains your industrial textile/garment data with sample/train/good and sample/test/defect_type structure.

Required Internal Structure (Example: MVTec AD bottle class):

```text
data/
├── dataset_mvtec/
│   ├── bottle/
│   │   ├── train/
│   │   │   └── good/
│   │   │       ├── 000.png
│   │   │       └── ... (10 images selected for training)
│   │   └── test/
│   │       ├── good/
│   │       ├── screw_up/  # NG
│   │       └── crack/     # NG
└── ...

4. Model Weights
The feature extractor, EfficientNet-B7, must be available locally. Update CONFIG["MODEL_PATH"] in main.py to point to the local cache directory of your EfficientNet-B7 model weights, typically loaded from the Hugging Face hub.
⚙️ Running the Analysis
The core logic resides in main.py. The script will automatically:

Load the model and set the device to CUDA (if available).
Iterate through all dataset folders found in data/.
Cache feature vectors for speed.
Run all 5 anomaly detection strategies across all samples with the specified hyperparameter grid.
Generate performance reports, statistical analysis, and visualizations.

python main.py

Output
Results will be organized under the results/ directory:

``` text
results/
├── dataset_mvtec/
│   ├── report_dataset_mvtec.xlsx  # Final performance summary (NG F1, OK F1, Time)
│   ├── charts/                    # Heatmaps and Bar Charts
│   └── showcases/                 # Qualitative image results (TP, FP, FN, TN)
├── dataset_proprietary/
└── ...

📝 Reproducibility and Licensing

Random Seed: All random operations (e.g., initial clustering, t-SNE, VAE/AnoGAN weights) use a fixed seed of 42.
Licensing: This project is licensed under the MIT License.
