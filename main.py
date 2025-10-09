# file: main.py
# DESCRIPTION:
# - Phiên bản cuối cùng, đã sửa lỗi `ValueError: Input contains NaN.` bằng cách xử lý an toàn.
# - Tự động hóa hoàn toàn, xử lý nhiều bộ dữ liệu, tạo đầy đủ báo cáo và visualizations.
# - Sửa sắp xếp sample_names theo thứ tự số học (numerical sort).
# - Cập nhật generate_showcase_reports để tạo showcases cho đủ 5 phương pháp (chọn sample NG_F1 max cho mỗi method).

import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from itertools import product
import re # << THÊM VÀO: Thư viện cần thiết cho việc sắp xếp tự nhiên

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. CONFIGURATION ---
CONFIG = {
    "DATASET_ROOT": "data",
    "MODEL_PATH": r"D:\scr\journal\models--google--efficientnet-b7\snapshots\7fb0338488c493e554df736b442d7d3ae43770ca", # Tự động tải model
    "OUTPUT_DIR_ROOT": "results",
    "DPI": 400,
    "HYPERPARAMS": {
        "KMeans_Multi_Centroid": {"n_clusters": [2, 3, 4]},
        "One_Class_SVM": {"nu": [0.05, 0.1, 0.2], "gamma": ['scale', 'auto']},
        "Single_Centroid": {},
        "VAE": {"epochs": [50], "latent_dim": [128]},
        "AnoGAN": {"epochs": [50], "latent_dim": [128], "inference_steps": [100], "w_adv": [0.1], "w_rec": [0.9]},
    },
    "STRICTNESS_LEVELS": np.arange(1.5, 0.75, -0.05).tolist(),
}

# --- 2. DEEP LEARNING MODEL DEFINITIONS ---
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=128):
        super(VAE, self).__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2)); self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def reparameterize(self, mu, logvar): std = torch.exp(0.5 * logvar); return mu + torch.randn_like(std) * std
    def forward(self, x): h = self.encoder(x); mu, logvar = torch.chunk(h, 2, dim=-1); z = self.reparameterize(mu, logvar); return self.decoder(z), mu, logvar
class Generator(nn.Module):
    def __init__(self, output_dim, latent_dim=128): super().__init__(); self.model = nn.Sequential(nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2), nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.Linear(512, output_dim))
    def forward(self, z): return self.model(z)
class Discriminator(nn.Module):
    def __init__(self, input_dim): super().__init__(); self.model = nn.Sequential(nn.Linear(input_dim, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x): return self.model(x).view(-1, 1).squeeze(1)

# --- 3. CORE FUNCTIONS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
def load_model_and_processor(model_path):
    try: processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True); model = AutoModel.from_pretrained(model_path).to(device); model.eval(); return processor, model
    except Exception as e: print(f"Error loading model: {e}"); return None, None
def extract_feature(image_path, processor, model):
    try: 
        image = Image.open(image_path).convert("RGB"); inputs = processor(images=image, return_tensors="pt").to(device);
        with torch.no_grad(): outputs = model(**inputs); return outputs.pooler_output.squeeze().cpu().numpy()
    except Exception: return None

def prepare_dataset_structure(root_folder, dataset_name):
    dataset, labels = {}, {}
    dataset_path = os.path.join(root_folder, dataset_name)
    try:
        # --- SỬA LỖI SẮP XẾP TẠI ĐÂY ---
        # Sử dụng "natural sort" để đảm bảo các tên như '10' đứng sau '2'.
        # Logic này tách các phần chữ và số trong tên để so sánh một cách chính xác.
        sample_names = sorted(
            [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))],
            key=lambda s: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]
        )
    except FileNotFoundError: print(f"ERROR: Dataset '{dataset_name}' not found"); return {}, {}, []
    for name in sample_names:
        train_good_path = os.path.join(dataset_path, name, 'train', 'good'); test_path = os.path.join(dataset_path, name, 'test')
        if not (os.path.isdir(train_good_path) and os.path.isdir(test_path)): continue
        train_files = [os.path.join(train_good_path, f) for f in os.listdir(train_good_path)]; test_files, defect_types = [], set()
        for subfolder in os.listdir(test_path):
            sub_path = os.path.join(test_path, subfolder);
            if os.path.isdir(sub_path):
                folder_name_lower = subfolder.lower()
                is_good = folder_name_lower == 'good' or folder_name_lower == 'ok'
                label = 'OK' if is_good else 'NG'
                if not is_good: defect_types.add(subfolder)
                for f in os.listdir(sub_path): file_path = os.path.join(sub_path, f); test_files.append(file_path); labels[file_path] = label
        if defect_types: print(f"  - Sample '{name}': Found NG types {sorted(list(defect_types))}, mapped to 'NG'.")
        dataset[name] = {'train_files': sorted(train_files), 'test_files': sorted(test_files)}
    print(f"Found {len(dataset)} valid samples for '{dataset_name}': {list(dataset.keys())}"); return dataset, labels, list(dataset.keys())

# --- 4. ANALYSIS LOGIC ---
def analyze_single_centroid(train_vectors, test_files, features_cache, **kwargs):
    start_train = time.time(); centroid = np.mean(train_vectors, axis=0); base_distance = np.max(np.linalg.norm(train_vectors - centroid, axis=1)); train_time = time.time() - start_train
    test_vectors = np.array([features_cache[f] for f in test_files]); start_inf = time.time(); scores = np.linalg.norm(test_vectors - centroid, axis=1); inference_time = (time.time() - start_inf) / len(test_files) if test_files else 0
    results = [{'strictness': factor, 'predictions': ["OK" if s <= base_distance * factor else "NG" for s in scores], 'scores': scores} for factor in CONFIG["STRICTNESS_LEVELS"]]; return results, train_time, inference_time
def analyze_kmeans_multicentroid(train_vectors, test_files, features_cache, n_clusters=3):
    if len(train_vectors) <= n_clusters: return [], 0, 0
    start_train = time.time(); kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(train_vectors); centroids = kmeans.cluster_centers_; base_distances = [np.max(np.linalg.norm(train_vectors[kmeans.labels_ == i] - centroids[i], axis=1)) if np.any(kmeans.labels_ == i) else 0 for i in range(n_clusters)]; train_time = time.time() - start_train
    test_vectors = np.array([features_cache[f] for f in test_files]); start_inf = time.time(); distances_matrix = kmeans.transform(test_vectors); scores = np.min(distances_matrix, axis=1); inference_time = (time.time() - start_inf) / len(test_files) if test_files else 0; results = []
    for factor in CONFIG["STRICTNESS_LEVELS"]: predictions = ["OK" if score <= base_distances[np.argmin(distances_matrix[i])] * factor else "NG" for i, score in enumerate(scores)]; results.append({'strictness': factor, 'predictions': predictions, 'scores': scores}); return results, train_time, inference_time
def analyze_one_class_svm(train_vectors, test_files, features_cache, nu=0.1, gamma='scale'):
    start_train = time.time(); svm = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma).fit(train_vectors); train_time = time.time() - start_train
    test_vectors = np.array([features_cache[f] for f in test_files]); start_inf = time.time(); scores = -svm.decision_function(test_vectors); inference_time = (time.time() - start_inf) / len(test_files) if test_files else 0; base_threshold = -np.min(svm.decision_function(train_vectors))
    results = [{'strictness': factor, 'predictions': ["OK" if s <= base_threshold * factor else "NG" for s in scores], 'scores': scores} for factor in CONFIG["STRICTNESS_LEVELS"]]; return results, train_time, inference_time
def analyze_vae(train_vectors, test_files, features_cache, epochs=50, latent_dim=128):
    input_dim = train_vectors.shape[1]; scaler = StandardScaler(); train_vectors_scaled = scaler.fit_transform(train_vectors)
    def vae_loss(recon, x, mu, logvar): return nn.functional.mse_loss(recon, x, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    start_train = time.time(); train_loader = DataLoader(TensorDataset(torch.tensor(train_vectors_scaled, dtype=torch.float32)), batch_size=min(16, len(train_vectors)), shuffle=True); model = VAE(input_dim, latent_dim=latent_dim).to(device); optimizer = optim.Adam(model.parameters(), lr=1e-3); model.train()
    for _ in range(epochs):
        for data in train_loader: x = data[0].to(device); recon, mu, logvar = model(x); loss = vae_loss(recon, x, mu, logvar); optimizer.zero_grad(); loss.backward(); optimizer.step()
    train_time = time.time() - start_train; model.eval()
    with torch.no_grad(): recon_train, _, _ = model(torch.tensor(train_vectors_scaled, dtype=torch.float32).to(device)); train_errors = nn.functional.mse_loss(recon_train, torch.tensor(train_vectors_scaled, dtype=torch.float32).to(device), reduction='none').mean(axis=1).cpu().numpy(); base_threshold = np.percentile(train_errors, 95)
    test_vectors = np.array([features_cache[f] for f in test_files]); test_vectors_scaled = scaler.transform(test_vectors); start_inf = time.time()
    with torch.no_grad(): recon_test, _, _ = model(torch.tensor(test_vectors_scaled, dtype=torch.float32).to(device)); scores = nn.functional.mse_loss(recon_test, torch.tensor(test_vectors_scaled, dtype=torch.float32).to(device), reduction='none').mean(axis=1).cpu().numpy()
    inference_time = (time.time() - start_inf) / len(test_files) if test_files else 0
    results = [{'strictness': factor, 'predictions': ["OK" if s <= base_threshold * factor else "NG" for s in scores], 'scores': scores} for factor in CONFIG["STRICTNESS_LEVELS"]]; return results, train_time, inference_time
def analyze_anogan(train_vectors, test_files, features_cache, epochs=50, latent_dim=128, inference_steps=100, w_adv=0.1, w_rec=0.9):
    input_dim = train_vectors.shape[1]; scaler = StandardScaler(); train_vectors_scaled = scaler.fit_transform(train_vectors)
    start_train = time.time(); g = Generator(input_dim, latent_dim).to(device); d = Discriminator(input_dim).to(device); optim_d = optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999)); optim_g = optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999)); criterion = nn.BCELoss()
    train_loader = DataLoader(TensorDataset(torch.tensor(train_vectors_scaled, dtype=torch.float32)), batch_size=min(16, len(train_vectors)), shuffle=True)
    for _ in range(epochs):
        for data in train_loader:
            real = data[0].to(device); batch_size = real.size(0); valid = torch.ones(batch_size, device=device); fake = torch.zeros(batch_size, device=device)
            optim_d.zero_grad(); d_loss = (criterion(d(real), valid) + criterion(d(g(torch.randn(batch_size, latent_dim, device=device)).detach()), fake)) / 2; d_loss.backward(); optim_d.step()
            optim_g.zero_grad(); g_loss = criterion(d(g(torch.randn(batch_size, latent_dim, device=device))), valid); g_loss.backward(); optim_g.step()
    train_time = time.time() - start_train; g.eval(); d.eval()
    train_scores = []
    for vec in train_vectors_scaled:
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device); z = torch.randn(1, latent_dim, device=device, requires_grad=True); z_optimizer = optim.Adam([z], lr=1e-3)
        for _ in range(inference_steps): loss = w_rec * nn.functional.mse_loss(g(z), x) + w_adv * nn.functional.mse_loss(d(g(z)), d(x)); z_optimizer.zero_grad(); loss.backward(); z_optimizer.step()
        train_scores.append(loss.item())
    base_threshold = np.percentile(train_scores, 95); scores = []; test_vectors = np.array([features_cache[f] for f in test_files]); test_vectors_scaled = scaler.transform(test_vectors); start_inf = time.time()
    for vec in test_vectors_scaled:
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device); z = torch.randn(1, latent_dim, device=device, requires_grad=True); z_optimizer = optim.Adam([z], lr=1e-3)
        for _ in range(inference_steps): loss = w_rec * nn.functional.mse_loss(g(z), x) + w_adv * nn.functional.mse_loss(d(g(z)), d(x)); z_optimizer.zero_grad(); loss.backward(); z_optimizer.step()
        scores.append(loss.item())
    inference_time = (time.time() - start_inf) / len(test_files) if test_files else 0
    results = [{'strictness': factor, 'predictions': ["OK" if s <= base_threshold * factor else "NG" for s in scores], 'scores': scores} for factor in CONFIG["STRICTNESS_LEVELS"]]; return results, train_time, inference_time

# --- 5. REPORTING & VISUALIZATION FUNCTIONS ---
def generate_final_report(all_runs_df, output_dir, sample_order, dataset_name):
    if all_runs_df.empty: return None
    print(f"\n--- GENERATING COMPREHENSIVE REPORT for '{dataset_name}' ---"); charts_dir = os.path.join(output_dir, "charts"); os.makedirs(charts_dir, exist_ok=True); report_file = os.path.join(output_dir, f"report_{dataset_name}.xlsx")
    
    # --- SỬA LỖI TẠI ĐÂY ---
    def calculate_metrics(g):
        y_true, y_pred, scores = g['true_label'], g['prediction'], g['score']
        y_true_binary = (y_true == 'NG').astype(int)
        scores_np = np.asarray(scores)
        
        # Kiểm tra điều kiện không thể tính AUROC/AUPR:
        # 1. Chỉ có 1 lớp (OK hoặc NG) trong dữ liệu test.
        # 2. Mảng scores chứa giá trị NaN hoặc vô cực (do mô hình bất ổn).
        if len(np.unique(y_true_binary)) < 2 or not np.all(np.isfinite(scores_np)):
            auroc, aupr = np.nan, np.nan
        else:
            auroc = roc_auc_score(y_true_binary, scores_np)
            aupr = average_precision_score(y_true_binary, scores_np)
            
        return pd.Series({'OK_F1': f1_score(y_true, y_pred, pos_label='OK', zero_division=0), 'NG_F1': f1_score(y_true, y_pred, pos_label='NG', zero_division=0), 'AUROC': auroc, 'AUPR': aupr})

    metrics_df = all_runs_df.groupby(['sample', 'method', 'params', 'strictness']).apply(calculate_metrics).reset_index(); metrics_df['F1_Sum'] = metrics_df['OK_F1'] + metrics_df['NG_F1']
    best_strictness_df = metrics_df.loc[metrics_df.groupby(['sample', 'method', 'params'])['F1_Sum'].idxmax()]; final_summary_df = best_strictness_df.loc[best_strictness_df.groupby(['sample', 'method'])['F1_Sum'].idxmax()]
    time_info = all_runs_df[['sample', 'method', 'params', 'train_time', 'inference_time_per_image_ms']].drop_duplicates()
    final_summary_df = pd.merge(final_summary_df, time_info, on=['sample', 'method', 'params']).rename(columns={'strictness': 'best_strictness', 'params': 'best_params'})
    final_summary_df['sample'] = pd.Categorical(final_summary_df['sample'], categories=sample_order, ordered=True); final_summary_df = final_summary_df.sort_values('sample').reset_index(drop=True)
    with pd.ExcelWriter(report_file, engine='xlsxwriter') as writer: final_summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
    print(f"Excel report saved to: '{report_file}'")
    for metric in ['NG_F1', 'OK_F1', 'AUROC', 'AUPR', 'train_time', 'inference_time_per_image_ms']:
        is_score_metric = any(x in metric for x in ['F1', 'AU']); pivot_table = final_summary_df.pivot(index='sample', columns='method', values=metric)
        plt.figure(figsize=(max(12, len(sample_order)*0.8), max(8, len(final_summary_df['method'].unique())*0.6))); sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis" if is_score_metric else "rocket_r", linewidths=.5, vmin=0.0 if is_score_metric else None, vmax=1.0 if is_score_metric else None)
        plt.title(f'Heatmap: {metric} - Dataset: {dataset_name}', fontsize=16); plt.xlabel('Method'); plt.ylabel('Sample'); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(os.path.join(charts_dir, f"heatmap_{metric}.png"), dpi=CONFIG["DPI"]); plt.close()
        if is_score_metric:
            pivot_table.plot(kind='bar', figsize=(max(16, len(sample_order)*1.2), 8), width=0.8); plt.title(f'Bar Chart: {metric} - Dataset: {dataset_name}', fontsize=16); plt.ylabel(metric); plt.xlabel('Sample'); plt.xticks(rotation=45, ha="right"); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.legend(title='Method'); plt.tight_layout(); plt.savefig(os.path.join(charts_dir, f"barchart_{metric}.png"), dpi=CONFIG["DPI"]); plt.close()
    print(f"Charts and Heatmaps saved to '{charts_dir}'"); return final_summary_df

def generate_tsne_visualization(dataset, features_cache, output_dir, dataset_name):
    print("\n--- GENERATING T-SNE VISUALIZATION ---"); vis_dir = os.path.join(output_dir, "visualizations"); os.makedirs(vis_dir, exist_ok=True)
    variances = {name: np.mean(np.var(np.array([features_cache[f] for f in paths['train_files'] if features_cache.get(f) is not None]), axis=0)) for name, paths in dataset.items() if len(paths['train_files']) > 1}
    if not variances: print("Could not calculate variances for t-SNE."); return
    high_consistency_sample, high_variance_sample = min(variances, key=variances.get), max(variances, key=variances.get)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)); fig.suptitle(f't-SNE Visualization - Dataset: {dataset_name}', fontsize=24)
    for ax, (sample_name, title) in zip(axes, [(high_consistency_sample, "High-Consistency"), (high_variance_sample, "High-Variance")]):
        paths = dataset[sample_name]; vecs = {label: [] for label in ['Train_OK', 'Test_OK', 'Test_NG']}
        for f in paths['train_files']: vecs['Train_OK'].append(features_cache.get(f))
        for f in paths['test_files']:
            folder_name = os.path.basename(os.path.dirname(f)).lower()
            label = 'Test_OK' if folder_name == 'good' or folder_name == 'ok' else 'Test_NG'
            vecs[label].append(features_cache.get(f))
        all_vecs = np.vstack([v for k in ['Train_OK', 'Test_NG', 'Test_OK'] for v in vecs[k] if v is not None]); perplexity = min(30, max(1, all_vecs.shape[0] - 1))
        tsne_results = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(all_vecs); n_train = len([v for v in vecs['Train_OK'] if v is not None]); n_ng = len([v for v in vecs['Test_NG'] if v is not None])
        ax.scatter(tsne_results[:n_train, 0], tsne_results[:n_train, 1], c='blue', label='Train (OK)', s=80, alpha=0.8); ax.scatter(tsne_results[n_train:n_train+n_ng, 0], tsne_results[n_train:n_train+n_ng, 1], c='red', label='Test (NG)', s=80, alpha=0.8); ax.scatter(tsne_results[n_train+n_ng:, 0], tsne_results[n_train+n_ng:, 1], c='green', label='Test (OK)', s=80, alpha=0.8)
        ax.set_title(f"Sample: {sample_name} ({title})"); ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); output_path = os.path.join(vis_dir, "tsne_visualization.png"); plt.savefig(output_path, dpi=CONFIG["DPI"]); plt.close(); print(f"t-SNE visualization saved to: '{output_path}'")

def generate_showcase_reports(performance_df, all_results_df, output_dir, dataset_name):
    print("\n--- GENERATING DYNAMIC SHOWCASE REPORTS ---"); showcase_dir = os.path.join(output_dir, "showcases"); os.makedirs(showcase_dir, exist_ok=True)
    def plot_qualitative(sample, method, prefix):
        df = all_results_df[(all_results_df['sample'] == sample) & (all_results_df['method'] == method)]
        if df.empty: return
        cases = {'TP': df[(df['true_label'] == 'NG') & (df['prediction'] == 'NG')], 'FN': df[(df['true_label'] == 'NG') & (df['prediction'] == 'OK')], 'FP': df[(df['true_label'] == 'OK') & (df['prediction'] == 'NG')], 'TN': df[(df['true_label'] == 'OK') & (df['prediction'] == 'OK')]}
        fig, axes = plt.subplots(2, 2, figsize=(15, 15)); fig.suptitle(f'Showcase: {sample} ({method}) - Dataset: {dataset_name}', fontsize=24)
        titles = {"TP": "True Positive (Defect Detected)", "FN": "False Negative (Defect Missed)", "FP": "False Positive (False Alarm)", "TN": "True Negative (Normal Identified)"}
        for ax, (case_type, case_df) in zip(axes.flatten(), cases.items()):
            ax.set_title(titles[case_type]); ax.set_xticks([]); ax.set_yticks([])
            if not case_df.empty: ax.imshow(Image.open(case_df['image_path'].iloc[0]).convert('RGB'))
            else: ax.text(0.5, 0.5, "No example found", ha='center', va='center')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(showcase_dir, f"{prefix}{sample}_{method.replace(' ','_')}.png"), dpi=CONFIG["DPI"]); plt.close()

    # Case gốc: Failure cho VAE
    try: vae_results = performance_df[performance_df['method'] == 'VAE']; fail_case = vae_results.loc[vae_results['NG_F1'].idxmin()]; print(f"Found 'Critical Failure' case: VAE on {fail_case['sample']} (NG_F1: {fail_case['NG_F1']:.2f}, lowest for this dataset)"); plot_qualitative(fail_case['sample'], 'VAE', 'showcase_failure_')
    except (IndexError, ValueError): print("INFO: Could not find a 'Critical Failure' case for VAE.")
    
    # Case gốc: Success cho method có improvement lớn
    try:
        adv_methods = ['KMeans_Multi_Centroid', 'One_Class_SVM', 'AnoGAN', 'VAE']; df_merged = pd.merge(performance_df[performance_df['method'].isin(adv_methods)], performance_df[performance_df['method'] == 'Single_Centroid'][['sample', 'NG_F1']], on='sample', suffixes=('', '_baseline')); df_merged['improvement'] = df_merged['NG_F1'] - df_merged['NG_F1_baseline']; improvements_only = df_merged[df_merged['improvement'] > 0.01]
        if improvements_only.empty: raise IndexError
        success_case = improvements_only.loc[improvements_only['improvement'].idxmax()]; print(f"Found 'Illustrative Success' case: {success_case['method']} on {success_case['sample']} (Largest improvement: +{success_case['improvement']:.2f})"); plot_qualitative(success_case['sample'], success_case['method'], 'showcase_success_')
    except (IndexError, KeyError, ValueError): print("INFO: Could not find a clear 'Illustrative Success' case.")
    
    # Case gốc: Trade-off cho KMeans vs Single_Centroid
    try:
        pivot_df = performance_df.pivot_table(index='sample', columns='method', values=['OK_F1', 'NG_F1']); ng_improvement = pivot_df[('NG_F1', 'KMeans_Multi_Centroid')] - pivot_df[('NG_F1', 'Single_Centroid')]; ok_loss = pivot_df[('OK_F1', 'Single_Centroid')] - pivot_df[('OK_F1', 'KMeans_Multi_Centroid')]; tradeoff_df = pd.DataFrame({'ng_improvement': ng_improvement, 'ok_loss': ok_loss}).dropna(); tradeoff_candidates = tradeoff_df[(tradeoff_df['ng_improvement'] > 0) & (tradeoff_df['ok_loss'] > 0)]
        if tradeoff_candidates.empty: raise IndexError
        tradeoff_sample = tradeoff_candidates['ng_improvement'].idxmax(); print(f"Found 'Industrial Trade-Off' case on sample: {tradeoff_sample}"); plot_qualitative(tradeoff_sample, 'Single_Centroid', 'showcase_tradeoff_baseline_'); plot_qualitative(tradeoff_sample, 'KMeans_Multi_Centroid', 'showcase_tradeoff_complex_')
    except (IndexError, KeyError, ValueError): print("INFO: Could not find a clear 'Industrial Trade-Off' case.")
    
    # Thêm: Tạo showcase cho từng method unique (5 method), chọn sample có NG_F1 cao nhất làm tiêu biểu
    methods = performance_df['method'].unique()  # Lấy tất cả method (nên có 5)
    for method in methods:
        try:
            method_results = performance_df[performance_df['method'] == method]
            if not method_results.empty:
                best_sample = method_results.loc[method_results['NG_F1'].idxmax()]['sample']
                print(f"Creating showcase for {method} on best sample: {best_sample} (NG_F1: {method_results['NG_F1'].max():.2f})")
                plot_qualitative(best_sample, method, 'showcase_success_sample_')
        except (IndexError, ValueError): print(f"INFO: Could not create showcase for {method}.")
    
    print(f"Showcase reports saved to '{showcase_dir}'")

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    all_datasets = [d for d in os.listdir(CONFIG["DATASET_ROOT"]) if os.path.isdir(os.path.join(CONFIG["DATASET_ROOT"], d))]
    if not all_datasets: print(f"No datasets found in '{CONFIG['DATASET_ROOT']}'. Exiting."); exit()
    print(f"Found {len(all_datasets)} datasets to process: {all_datasets}")
    
    print("\nLoading model and processor once..."); processor, model = load_model_and_processor(CONFIG["MODEL_PATH"])
    if model is None: exit()
    features_cache = {}

    for dataset_name in all_datasets:
        print(f"\n{'#'*25} STARTING ANALYSIS FOR DATASET: {dataset_name.upper()} {'#'*25}")
        output_dir = os.path.join(CONFIG["OUTPUT_DIR_ROOT"], dataset_name); os.makedirs(output_dir, exist_ok=True)
        dataset, labels, sample_order = prepare_dataset_structure(CONFIG["DATASET_ROOT"], dataset_name)
        if not dataset: print(f"Skipping dataset '{dataset_name}'."); continue
        
        all_files_current = [img for s in dataset.values() for f in ['train_files', 'test_files'] for img in s[f]]
        for img_path in tqdm(all_files_current, desc=f"Caching Features for {dataset_name}"):
            if img_path not in features_cache: features_cache[img_path] = extract_feature(img_path, processor, model)
        
        strategies = {"Single_Centroid": analyze_single_centroid, "KMeans_Multi_Centroid": analyze_kmeans_multicentroid, "One_Class_SVM": analyze_one_class_svm, "VAE": analyze_vae, "AnoGAN": analyze_anogan}
        master_results_list = []
        
        for method_name, analysis_func in strategies.items():
            param_grid = CONFIG["HYPERPARAMS"].get(method_name, {}); keys, values = zip(*param_grid.items()) if param_grid else ([], [])
            param_combinations = [dict(zip(keys, v)) for v in product(*values)] if param_grid else [{}]
            print(f"\n{'='*20} RUNNING METHOD: {method_name.upper()} {'='*20}")
            for params in param_combinations:
                params_str = str(params) if params else "default"; tqdm_desc = f"Processing ({method_name} | {params_str})"
                for sample_name in tqdm(sample_order, desc=tqdm_desc):
                    paths = dataset.get(sample_name);
                    if not paths: continue
                    train_vectors = np.array([features_cache[f] for f in paths['train_files'] if features_cache.get(f) is not None])
                    valid_test_files = [f for f in paths['test_files'] if features_cache.get(f) is not None]
                    if len(train_vectors) < 2 or not valid_test_files: continue
                    run_results, train_time, inference_time = analysis_func(train_vectors, valid_test_files, features_cache, **params)
                    true_labels = [labels[f] for f in valid_test_files]
                    for res in run_results:
                        for i, pred in enumerate(res['predictions']): master_results_list.append({'sample': sample_name, 'method': method_name, 'params': params_str, 'strictness': res['strictness'], 'true_label': true_labels[i], 'prediction': pred, 'score': res['scores'][i], 'train_time': train_time, 'inference_time_per_image_ms': inference_time * 1000, 'image_path': valid_test_files[i]})
        
        if master_results_list:
            all_runs_df = pd.DataFrame(master_results_list)
            performance_summary_df = generate_final_report(all_runs_df, output_dir, sample_order, dataset_name)
            if performance_summary_df is not None:
                best_results_df = pd.merge(all_runs_df, performance_summary_df[['sample', 'method', 'best_strictness']], left_on=['sample', 'method', 'strictness'], right_on=['sample', 'method', 'best_strictness'], how='inner')
                generate_tsne_visualization(dataset, features_cache, output_dir, dataset_name)
                generate_showcase_reports(performance_summary_df, best_results_df, output_dir, dataset_name)
        else:
            print(f"No results were generated for dataset '{dataset_name}'.")
        print(f"\n--- ANALYSIS COMPLETE FOR DATASET: {dataset_name.upper()} ---")

    print(f"\n{'#'*20} ALL DATASETS PROCESSED {'#'*20}")