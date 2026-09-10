import time
import sys
import os
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType

from src.data_loader import load_and_split_data
from src.baseline_als import BaselineALS
from src.short_term_model import ShortTermModel
from src.weighting import compute_hybrid_weights
from src.pareto_optimizer import select_pareto_knee

SECONDS_PER_DAY = 86400
FIXED_TIME_INTERVAL = 7 * SECONDS_PER_DAY


class MetricTracker:
    """Helper class to accumulate evaluation metrics (Ranking & Rating Errors)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.ranking_records = []
        self.sq_errors = []
        self.abs_errors = []

    def add_sample(self, rec_items, gt_items, rating_pairs=None):
        sample_metrics = {}
        for k in [10, 50, 100]:
            sub_recs = rec_items[:k]
            hits = sum(1 for item in sub_recs if item in gt_items)

            sample_metrics[f'p@{k}'] = hits / k
            sample_metrics[f'r@{k}'] = hits / len(gt_items) if len(gt_items) > 0 else 0.0

            dcg = 0.0
            for idx, item in enumerate(sub_recs):
                if item in gt_items:
                    dcg += 1.0 / np.log2(idx + 2)
            idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(len(gt_items), k)))
            sample_metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0

        self.ranking_records.append(sample_metrics)

        if rating_pairs:
            for actual, pred in rating_pairs:
                err = actual - pred
                self.sq_errors.append(err ** 2)
                self.abs_errors.append(abs(err))

    def get_summary(self):
        if not self.ranking_records:
            return {}

        summary = {
            'rmse': np.sqrt(np.mean(self.sq_errors)) if self.sq_errors else 0.0,
            'mae': np.mean(self.abs_errors) if self.abs_errors else 0.0
        }
        for k in [10, 50, 100]:
            summary[f'ndcg@{k}'] = np.mean([r[f'ndcg@{k}'] for r in self.ranking_records])
            summary[f'p@{k}'] = np.mean([r[f'p@{k}'] for r in self.ranking_records])
            summary[f'r@{k}'] = np.mean([r[f'r@{k}'] for r in self.ranking_records])
        return summary


def extract_stream_dict(df_rows):
    user_stream = {}
    for r in df_rows:
        u, i, rating, ts = r['userId'], r['movieId'], r['rating'], r['timestamp']
        if u not in user_stream:
            user_stream[u] = []
        user_stream[u].append((i, rating, ts))
    return user_stream


def get_fast_top_k_short(centroid, item_matrix, item_id_arr, k=100):
    scores = np.dot(item_matrix, centroid)
    top_k_idx = np.argpartition(scores, -k)[-k:]
    sorted_top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
    top_k_items = item_id_arr[sorted_top_k_idx].tolist()
    score_dict = dict(zip(top_k_items, scores[sorted_top_k_idx]))
    return top_k_items, score_dict


def update_lookup_matrices(als_model):
    item_ids = list(als_model.item_factors.keys())
    item_id_arr = np.array(item_ids)
    item_matrix = np.array([als_model.item_factors[i] for i in item_ids])
    return item_ids, item_id_arr, item_matrix


def sanitize_df(df):
    return df.select(
        F.col("userId").cast(IntegerType()),
        F.col("movieId").cast(IntegerType()),
        F.col("rating").cast(FloatType()),
        F.col("timestamp").cast(IntegerType())
    ).dropna()


def print_checkpoint_metrics(model_name, retrain_num, week_num, total_weeks, metrics):
    print(f"\n" + "-" * 75)
    print(f" [{model_name}] Pre-Retrain #{retrain_num} Checkpoint (Week {week_num}/{total_weeks})")
    print("-" * 75)
    print(f"  RMSE : {metrics['rmse']:.4f} | MAE : {metrics['mae']:.4f}")
    print(f"  NDCG @10 : {metrics['ndcg@10']:.4f} | @50 : {metrics['ndcg@50']:.4f} | @100 : {metrics['ndcg@100']:.4f}")
    print(f"  Prec @10 : {metrics['p@10']:.4f} | @50 : {metrics['p@50']:.4f} | @100 : {metrics['p@100']:.4f}")
    print(f"  Rec  @10 : {metrics['r@10']:.4f} | @50 : {metrics['r@50']:.4f} | @100 : {metrics['r@100']:.4f}")
    print("-" * 75)


def save_summary_table_file(m1_all, m2_all, s1_retrain_count, s2_retrain_count, s1_time, s2_time, best_params=None, output_dir="results_plots"):
    """Writes overall evaluation summary table to disk with 3D optimal parameters."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "evaluation_summary.txt")

    tau_str = f"{best_params['tau']:.4f}" if best_params else "N/A"
    gamma_str = f"{best_params['gamma']:.2f}" if best_params else "N/A"
    lambda_str = f"{best_params['lambda']:.2f}" if best_params else "N/A"

    content = (
        "\n" + "=" * 95 + "\n"
        "                      OVERALL MODEL PERFORMANCE SUMMARY (FULL TEST SET)\n"
        + "=" * 95 + "\n"
        f"{'Metric':<25} | {'Baseline ALS (7-Day Physical)':<32} | {'Hybrid (3D Pareto Opt)':<32}\n"
        + "-" * 95 + "\n"
        f"{'Optimal Tau (tau*)':<25} | {'N/A':<32} | {tau_str:<32}\n"
        f"{'Optimal Gamma (gamma*)':<25} | {'N/A':<32} | {gamma_str:<32}\n"
        f"{'Optimal Lambda (lambda*)':<25} | {'N/A':<32} | {lambda_str:<32}\n"
        f"{'Total Retrains':<25} | {s1_retrain_count:<32} | {s2_retrain_count:<32}\n"
        f"{'Execution Time':<25} | {s1_time:<32.2f}s | {s2_time:<32.2f}s\n"
        + "-" * 95 + "\n"
        f"{'RMSE':<25} | {m1_all.get('rmse',0):<32.4f} | {m2_all.get('rmse',0):<32.4f}\n"
        f"{'MAE':<25} | {m1_all.get('mae',0):<32.4f} | {m2_all.get('mae',0):<32.4f}\n"
        + "-" * 95 + "\n"
        f"{'NDCG @ 10':<25} | {m1_all.get('ndcg@10',0):<32.4f} | {m2_all.get('ndcg@10',0):<32.4f}\n"
        f"{'NDCG @ 50':<25} | {m1_all.get('ndcg@50',0):<32.4f} | {m2_all.get('ndcg@50',0):<32.4f}\n"
        f"{'NDCG @ 100':<25} | {m1_all.get('ndcg@100',0):<32.4f} | {m2_all.get('ndcg@100',0):<32.4f}\n"
        + "-" * 95 + "\n"
        f"{'Precision @ 10':<25} | {m1_all.get('p@10',0):<32.4f} | {m2_all.get('p@10',0):<32.4f}\n"
        f"{'Precision @ 50':<25} | {m1_all.get('p@50',0):<32.4f} | {m2_all.get('p@50',0):<32.4f}\n"
        f"{'Precision @ 100':<25} | {m1_all.get('p@100',0):<32.4f} | {m2_all.get('p@100',0):<32.4f}\n"
        + "-" * 95 + "\n"
        f"{'Recall @ 10':<25} | {m1_all.get('r@10',0):<32.4f} | {m2_all.get('r@10',0):<32.4f}\n"
        f"{'Recall @ 50':<25} | {m1_all.get('r@50',0):<32.4f} | {m2_all.get('r@50',0):<32.4f}\n"
        f"{'Recall @ 100':<25} | {m1_all.get('r@100',0):<32.4f} | {m2_all.get('r@100',0):<32.4f}\n"
        + "=" * 95 + "\n"
    )

    with open(file_path, "w") as f:
        f.write(content)

    return content


def plot_dataset_split_summary(split_stats, output_dir="results_plots"):
    os.makedirs(output_dir, exist_ok=True)
    splits = ['Train (70%)', 'Val (15%)', 'Test (15%)']
    counts = [split_stats['train']['count'], split_stats['val']['count'], split_stats['test']['count']]
    weeks = [split_stats['train']['weeks'], split_stats['val']['weeks'], split_stats['test']['weeks']]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = '#1f77b4'
    ax1.set_xlabel('Dataset Split', fontweight='bold')
    ax1.set_ylabel('Interaction Count', color=color, fontweight='bold')
    bars = ax1.bar(splits, counts, color=color, alpha=0.6, width=0.4, label='Interactions')
    ax1.tick_params(axis='y', labelcolor=color)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + max(counts)*0.01, f'{yval:,}', ha='center', va='bottom', fontsize=9)

    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Duration (Weeks)', color=color, fontweight='bold')
    lines = ax2.plot(splits, weeks, color=color, marker='o', linewidth=2.5, markersize=8, label='Weeks')
    ax2.tick_params(axis='y', labelcolor=color)

    for i, w in enumerate(weeks):
        ax2.annotate(f'{w} Wks', (splits[i], weeks[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color=color)

    plt.title('Dataset Split Breakdown: Interactions & Time Horizon', fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_dataset_split_summary.png"), dpi=300)
    plt.close()


def plot_pareto_selection(pareto_candidates, best_params, output_dir="results_plots"):
    os.makedirs(output_dir, exist_ok=True)
    ndcgs = [c['ndcg'] for c in pareto_candidates]
    retrains = [c['retrains'] for c in pareto_candidates]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(retrains, ndcgs, c='#1f77b4', alpha=0.6, edgecolors='none', s=40, label='Grid Candidates')

    best_retrains = best_params['retrains']
    best_ndcg = best_params['ndcg']
    ax.scatter([best_retrains], [best_ndcg], color='red', s=130, zorder=5, marker='*', 
               label=f"Selected Knee (tau*={best_params['tau']:.2f}, gamma*={best_params['gamma']:.2f}, lambda*={best_params['lambda']:.1f})")

    ax.set_xlabel('Total Physical Retrains Triggered', fontweight='bold')
    ax.set_ylabel('Validation NDCG Score', fontweight='bold')
    ax.set_title('Step 2 Validation: 3D Pareto Knee Optimization Frontier', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_pareto_threshold_selection.png"), dpi=300)
    plt.close()

def plot_parameter_sensitivities(pareto_candidates, best_params, output_dir="results_plots"):
    """
    Generates 3 individual 1D sensitivity slice graphs around the optimal knee point.
    """
    os.makedirs(output_dir, exist_ok=True)
    tau_star, gamma_star, lambda_star = best_params['tau'], best_params['gamma'], best_params['lambda']

    # 1. Threshold (tau) Sensitivity
    slice_tau = sorted([c for c in pareto_candidates if c['gamma'] == gamma_star and c['lambda'] == lambda_star], key=lambda x: x['tau'])
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel(f'Threshold tau (Fixed gamma*={gamma_star}, lambda*={lambda_star})', fontweight='bold')
    ax1.set_ylabel('Validation NDCG Score', color='#1f77b4', fontweight='bold')
    ax1.plot([c['tau'] for c in slice_tau], [c['ndcg'] for c in slice_tau], color='#1f77b4', marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Physical Retrains', color='#ff7f0e', fontweight='bold')
    ax2.plot([c['tau'] for c in slice_tau], [c['retrains'] for c in slice_tau], color='#ff7f0e', marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    ax1.axvline(x=tau_star, color='red', linestyle=':', linewidth=2)
    best_c = next(c for c in slice_tau if c['tau'] == tau_star)
    ax1.scatter([tau_star], [best_c['ndcg']], color='red', s=150, zorder=5, marker='*', label=f'Knee (tau*={tau_star:.2f})')
    ax1.legend(loc='center left')
    plt.title('Parameter Sensitivity: Threshold (tau)', fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "2a_tau_sensitivity.png"), dpi=300)
    plt.close()

    # 2. Decay Rate (gamma) Sensitivity
    slice_gamma = sorted([c for c in pareto_candidates if c['tau'] == tau_star and c['lambda'] == lambda_star], key=lambda x: x['gamma'])
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel(f'Decay Rate gamma (Fixed tau*={tau_star:.2f}, lambda*={lambda_star})', fontweight='bold')
    ax1.set_ylabel('Validation NDCG Score', color='#1f77b4', fontweight='bold')
    ax1.plot([c['gamma'] for c in slice_gamma], [c['ndcg'] for c in slice_gamma], color='#1f77b4', marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Physical Retrains', color='#ff7f0e', fontweight='bold')
    ax2.plot([c['gamma'] for c in slice_gamma], [c['retrains'] for c in slice_gamma], color='#ff7f0e', marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    ax1.axvline(x=gamma_star, color='red', linestyle=':', linewidth=2)
    best_c = next(c for c in slice_gamma if c['gamma'] == gamma_star)
    ax1.scatter([gamma_star], [best_c['ndcg']], color='red', s=150, zorder=5, marker='*', label=f'Knee (gamma*={gamma_star:.2f})')
    ax1.legend(loc='center left')
    plt.title('Parameter Sensitivity: Decay Rate (gamma)', fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "2b_gamma_sensitivity.png"), dpi=300)
    plt.close()

    # 3. Confidence Parameter (lambda) Sensitivity
    slice_lambda = sorted([c for c in pareto_candidates if c['tau'] == tau_star and c['gamma'] == gamma_star], key=lambda x: x['lambda'])
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel(f'Confidence Parameter lambda (Fixed tau*={tau_star:.2f}, gamma*={gamma_star})', fontweight='bold')
    ax1.set_ylabel('Validation NDCG Score', color='#1f77b4', fontweight='bold')
    ax1.plot([c['lambda'] for c in slice_lambda], [c['ndcg'] for c in slice_lambda], color='#1f77b4', marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Physical Retrains', color='#ff7f0e', fontweight='bold')
    ax2.plot([c['lambda'] for c in slice_lambda], [c['retrains'] for c in slice_lambda], color='#ff7f0e', marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    ax1.axvline(x=lambda_star, color='red', linestyle=':', linewidth=2)
    best_c = next(c for c in slice_lambda if c['lambda'] == lambda_star)
    ax1.scatter([lambda_star], [best_c['ndcg']], color='red', s=150, zorder=5, marker='*', label=f'Knee (lambda*={lambda_star:.1f})')
    ax1.legend(loc='center left')
    plt.title('Parameter Sensitivity: Confidence (lambda)', fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "2c_lambda_sensitivity.png"), dpi=300)
    plt.close()


def plot_weekly_test_performance(s1_history, s2_history, output_dir="results_plots"):
    os.makedirs(output_dir, exist_ok=True)
    weeks_s1 = [h['week'] for h in s1_history]
    ndcg_s1 = [h['metrics'].get('ndcg@10', 0) for h in s1_history]

    weeks_s2 = [h['week'] for h in s2_history]
    ndcg_s2 = [h['metrics'].get('ndcg@10', 0) for h in s2_history]
    retrains_s2 = [h['retrained'] for h in s2_history]

    plt.figure(figsize=(10, 5))
    plt.plot(weeks_s1, ndcg_s1, label='Baseline ALS (Weekly Retrain)', color='#1f77b4', linestyle='--', marker='o', alpha=0.7)
    plt.plot(weeks_s2, ndcg_s2, label='Hybrid Model (Selective Retrain)', color='#2ca02c', linewidth=2.5, marker='s')

    retrain_weeks = [w for w, r in zip(weeks_s2, retrains_s2) if r]
    retrain_ndcgs = [n for n, r in zip(ndcg_s2, retrains_s2) if r]
    if retrain_weeks:
        plt.scatter(retrain_weeks, retrain_ndcgs, color='red', s=120, zorder=5, label='Physical Retrain Event', marker='*')

    plt.xlabel('Test Week', fontweight='bold')
    plt.ylabel('NDCG@10 Score', fontweight='bold')
    plt.title('Test Phase: Weekly NDCG@10 Tracking & Retrain Triggers', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_weekly_test_ndcg_comparison.png"), dpi=300)
    plt.close()


def plot_overall_performance(m1_all, m2_all, s1_retrain_count, s2_retrain_count, s1_time, s2_time, output_dir="results_plots"):
    os.makedirs(output_dir, exist_ok=True)
    models = ['Baseline ALS\n(7-Day Fixed)', 'Hybrid Model\n(3D Pareto Opt)']

    # 1. Retrain Operations Triggered
    plt.figure(figsize=(6, 5))
    bars1 = plt.bar(models, [s1_retrain_count, s2_retrain_count], color=['#1f77b4', '#2ca02c'], width=0.4)
    plt.ylabel('Total Retrain Count', fontweight='bold')
    plt.title('Retrain Operations Triggered', fontweight='bold')
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height()/2.0, f'{int(bar.get_height())}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_1_retrain_operations.png"), dpi=300)
    plt.close()

    # 2. Computation Time Comparison
    plt.figure(figsize=(6, 5))
    bars2 = plt.bar(models, [s1_time, s2_time], color=['#1f77b4', '#2ca02c'], width=0.4)
    plt.ylabel('Total Execution Time (s)', fontweight='bold')
    plt.title('Computation Time Comparison', fontweight='bold')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.5, f'{bar.get_height():.1f}s', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_2_computation_time.png"), dpi=300)
    plt.close()

    # 3. Rating Accuracy (RMSE/MAE)
    plt.figure(figsize=(6, 5))
    x = np.arange(2)
    width = 0.25
    plt.bar(x - width/2, [m1_all.get('rmse', 0), m2_all.get('rmse', 0)], width, label='RMSE', color='#ff7f0e')
    plt.bar(x + width/2, [m1_all.get('mae', 0), m2_all.get('mae', 0)], width, label='MAE', color='#2ca02c')
    plt.xticks(x, models)
    plt.ylabel('Rating Error', fontweight='bold')
    plt.title('Rating Accuracy (Lower is Better)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_3_rating_accuracy_errors.png"), dpi=300)
    plt.close()

    # 4. Ranking Performance (NDCG Metrics)
    plt.figure(figsize=(6, 5))
    k_vals = ['@10', '@50', '@100']
    m1_ndcg = [m1_all.get('ndcg@10', 0), m1_all.get('ndcg@50', 0), m1_all.get('ndcg@100', 0)]
    m2_ndcg = [m2_all.get('ndcg@10', 0), m2_all.get('ndcg@50', 0), m2_all.get('ndcg@100', 0)]

    x = np.arange(len(k_vals))
    width = 0.35
    plt.bar(x - width/2, m1_ndcg, width, label='Baseline ALS', color='#1f77b4')
    plt.bar(x + width/2, m2_ndcg, width, label='Hybrid Model', color='#2ca02c')
    plt.xticks(x, k_vals)
    plt.ylabel('NDCG Score', fontweight='bold')
    plt.title('Ranking Performance across Cutoffs', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_4_ranking_ndcg_performance.png"), dpi=300)
    plt.close()


def main():
    spark = SparkSession.builder \
        .appName("Physical-Retraining-Recommender-Benchmark") \
        .config("spark.driver.memory", "4g") \
        .config("spark.master", "local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print("[1/4] Loading dataset (70% Train / 15% Val / 15% Test split)...")
    DATA_PATH = "data/ratings1.dat"
    train_df, val_df, test_df = load_and_split_data(spark, DATA_PATH)

    train_df = sanitize_df(train_df)
    val_df = sanitize_df(val_df)
    test_df = sanitize_df(test_df)

    def get_segment_weeks(df):
        bounds = df.agg(F.min("timestamp"), F.max("timestamp")).collect()[0]
        if bounds[0] is None or bounds[1] is None:
            return 0
        return int(np.ceil((bounds[1] - bounds[0]) / FIXED_TIME_INTERVAL))

    split_stats = {
        'train': {'count': train_df.count(), 'weeks': get_segment_weeks(train_df)},
        'val':   {'count': val_df.count(),   'weeks': get_segment_weeks(val_df)},
        'test':  {'count': test_df.count(),  'weeks': get_segment_weeks(test_df)}
    }

    print("\n" + "=" * 65)
    print("                 DATASET SEGMENT BREAKDOWN SUMMARY")
    print("=" * 65)
    print(f" Train Split : {split_stats['train']['count']:,} interactions | {split_stats['train']['weeks']} weeks")
    print(f" Val Split   : {split_stats['val']['count']:,} interactions | {split_stats['val']['weeks']} weeks")
    print(f" Test Split  : {split_stats['test']['count']:,} interactions | {split_stats['test']['weeks']} weeks")
    print("=" * 65 + "\n")

    plot_dataset_split_summary(split_stats)

    train_seen_map = train_df.groupBy("userId") \
        .agg(F.collect_set("movieId").alias("seen")) \
        .rdd.collectAsMap()

    # -------------------------------------------------------------
    # STEP 1: INITIAL ALS FIT ON 70% TRAIN SET
    # -------------------------------------------------------------
    max_iterations = 15
    print(f"\n[2/4] Training Baseline ALS Model on 70% Train Split...")
    als_model = BaselineALS(rank=32, max_iter=max_iterations, reg_param=0.05)
    als_model.fit(train_df)

    # -------------------------------------------------------------
    # STEP 2: 3D HYPERPARAMETER VALIDATION SWEEP (tau, gamma, lambda)
    # -------------------------------------------------------------
    print("[3/4] Optimizing 3D Hyperparameters (tau*, gamma*, lambda*) via Pareto Knee...")

    min_val_ts = val_df.agg(F.min("timestamp")).collect()[0][0]
    max_val_ts = val_df.agg(F.max("timestamp")).collect()[0][0]
    total_val_weeks = int(np.ceil((max_val_ts - min_val_ts) / FIXED_TIME_INTERVAL))

    val_stream_full = extract_stream_dict(val_df.collect())
    val_users_sample = [u for u in list(val_stream_full.keys())[:300] if u in als_model.user_factors]

    # Grid search candidate bounds
    candidate_gammas = [0.60, 0.70, 0.80, 0.90]
    candidate_lambdas = [2.0, 5.0, 8.0, 10.0]
    candidate_taus = np.arange(0.1, 0.9, 0.05)

    weekly_eval_data = []
    curr_val_train_df = train_df
    val_window_start = min_val_ts
    week_num = 0

    print(f"\n--- Phase 1/2: Fitting Weekly ALS & Sampling 3D Dynamic Responses (Weeks: {total_val_weeks}) ---")
    while val_window_start < max_val_ts:
        val_window_end = val_window_start + FIXED_TIME_INTERVAL
        val_chunk = val_df.filter((F.col("timestamp") >= val_window_start) & (F.col("timestamp") < val_window_end))
        week_num += 1

        if val_chunk.count() > 0:
            print(f" -> [Week {week_num}/{total_val_weeks}] Processing weekly drift across candidate gammas...")
            curr_val_train_df = curr_val_train_df.unionByName(val_chunk)

            retrained_als = BaselineALS(rank=32, max_iter=max_iterations, reg_param=0.05)
            retrained_als.fit(curr_val_train_df)

            chunk_stream = extract_stream_dict(val_chunk.sort("timestamp").collect())

            candidates, item_id_arr, item_matrix = update_lookup_matrices(als_model)
            short_model = ShortTermModel(als_model.item_factors)

            r_candidates, r_item_id_arr, r_item_matrix = update_lookup_matrices(retrained_als)
            r_short_model = ShortTermModel(retrained_als.item_factors)

            # Structure to hold metrics per gamma & lambda
            week_gamma_data = {g: {'w2_lambdas': {l: [] for l in candidate_lambdas}, 'ndcg_frozen': [], 'ndcg_retrained': []} for g in candidate_gammas}

            for u in val_users_sample:
                if u not in chunk_stream:
                    continue
                interactions = chunk_stream[u]
                seen_items = set(train_seen_map.get(u, []))

                session_buf = []
                base_scores = als_model.predict_user_scores(u, candidates)
                r_base_scores = retrained_als.predict_user_scores(u, r_candidates)

                for i, (item_id, rating, ts) in enumerate(interactions):
                    gt_items = {item_id for item_id, r, _ in interactions[i:] if r >= 3.0}
                    if not gt_items:
                        seen_items.add(item_id)
                        continue

                    unseen_base_scores = {k: v for k, v in base_scores.items() if k not in seen_items}
                    top_k_base_masked = sorted(unseen_base_scores, key=unseen_base_scores.get, reverse=True)[:10]

                    r_unseen_base_scores = {k: v for k, v in r_base_scores.items() if k not in seen_items}
                    r_top_k_base_masked = sorted(r_unseen_base_scores, key=r_unseen_base_scores.get, reverse=True)[:10]

                    for gamma in candidate_gammas:
                        if not session_buf:
                            top_k_hybrid_frozen = top_k_base_masked
                            top_k_hybrid_retrained = r_top_k_base_masked
                            dissim = 0.0
                        else:
                            x_tilde = short_model.compute_centroid(session_buf, decay_gamma=gamma)
                            top_k_short, short_scores = get_fast_top_k_short(x_tilde, item_matrix, item_id_arr, k=10)
                            top_k_short_masked = [x for x in top_k_short if x not in seen_items]

                            # Calculate dissimilarity under this gamma
                            _, _, dissim = compute_hybrid_weights(len(session_buf), top_k_base_masked, top_k_short_masked, lambda_param=5.0)

                            candidate_pool = (set(top_k_base_masked) | set(top_k_short_masked)) - seen_items
                            # Use default lambda 5.0 for validation top-k candidate generation
                            w1, w2, _ = compute_hybrid_weights(len(session_buf), top_k_base_masked, top_k_short_masked, lambda_param=5.0)
                            hybrid_scores = {
                                item: (w1 * base_scores.get(item, 0.0)) + (w2 * short_scores.get(item, 0.0))
                                for item in candidate_pool
                            }
                            top_k_hybrid_frozen = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:10]

                            r_x_tilde = r_short_model.compute_centroid(session_buf, decay_gamma=gamma)
                            r_top_k_short, r_short_scores = get_fast_top_k_short(r_x_tilde, r_item_matrix, r_item_id_arr, k=10)
                            r_top_k_short_masked = [x for x in r_top_k_short if x not in seen_items]

                            r_w1, r_w2, _ = compute_hybrid_weights(len(session_buf), r_top_k_base_masked, r_top_k_short_masked, lambda_param=5.0)
                            r_candidate_pool = (set(r_top_k_base_masked) | set(r_top_k_short_masked)) - seen_items
                            r_hybrid_scores = {
                                item: (r_w1 * r_base_scores.get(item, 0.0)) + (r_w2 * r_short_scores.get(item, 0.0))
                                for item in r_candidate_pool
                            }
                            top_k_hybrid_retrained = sorted(r_hybrid_scores, key=r_hybrid_scores.get, reverse=True)[:10]

                        # Calculate w2 across lambda candidates
                        for l_param in candidate_lambdas:
                            conf = len(session_buf) / (len(session_buf) + l_param) if session_buf else 0.0
                            w2_val = dissim * conf
                            week_gamma_data[gamma]['w2_lambdas'][l_param].append(w2_val)

                        dcg_f = sum(1.0 / np.log2(idx + 2) for idx, item in enumerate(top_k_hybrid_frozen[:10]) if item in gt_items)
                        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(len(gt_items), 10)))
                        week_gamma_data[gamma]['ndcg_frozen'].append(dcg_f / idcg if idcg > 0 else 0.0)

                        dcg_r = sum(1.0 / np.log2(idx + 2) for idx, item in enumerate(top_k_hybrid_retrained[:10]) if item in gt_items)
                        week_gamma_data[gamma]['ndcg_retrained'].append(dcg_r / idcg if idcg > 0 else 0.0)

                    session_buf.append((item_id, rating))
                    seen_items.add(item_id)

            weekly_eval_data.append(week_gamma_data)

        val_window_start = val_window_end

    # Phase 2: Pareto Grid Evaluation over (gamma, lambda, tau)
    pareto_candidates = []
    param_grid = list(itertools.product(candidate_gammas, candidate_lambdas, candidate_taus))
    print(f"\n--- Phase 2/2: Sweeping 3D Parameter Grid (Total Candidate Configurations: {len(param_grid)}) ---")

    for gamma, l_param, tau in param_grid:
        total_ndcg = []
        retrain_count = 0

        for week_data in weekly_eval_data:
            g_data = week_data[gamma]
            avg_w2 = float(np.mean(g_data['w2_lambdas'][l_param])) if g_data['w2_lambdas'][l_param] else 0.0

            if avg_w2 >= tau:
                retrain_count += 1
                total_ndcg.append(np.mean(g_data['ndcg_retrained']))
            else:
                total_ndcg.append(np.mean(g_data['ndcg_frozen']))

        pareto_candidates.append({
            'tau': float(tau),
            'gamma': float(gamma),
            'lambda': float(l_param),
            'ndcg': float(np.mean(total_ndcg)) if total_ndcg else 0.0,
            'retrains': retrain_count
        })

    best_params = select_pareto_knee(pareto_candidates)
    tau_star = best_params['tau']
    gamma_star = best_params['gamma']
    lambda_star = best_params['lambda']

    print(f"\n -> Selected 3D Pareto Knee Parameters:")
    print(f"    * Optimal Threshold (tau*)    : {tau_star:.2f}")
    print(f"    * Optimal Decay Rate (gamma*)  : {gamma_star:.2f}")
    print(f"    * Optimal Confidence (lambda*) : {lambda_star:.2f}\n")

    plot_pareto_selection(pareto_candidates, best_params)
    plot_parameter_sensitivities(pareto_candidates, best_params)

    # -------------------------------------------------------------
    # STEP 3: TEST EVALUATION WITH PHYSICAL RETRAINING (15% TEST SET)
    # -------------------------------------------------------------
    print("[4/4] Cross-checking Models with Physical Retraining on 15% Test Split...")

    cumulative_base_df = train_df.unionByName(val_df).cache()
    cumulative_base_df.count()

    min_test_ts = test_df.agg(F.min("timestamp")).collect()[0][0]
    max_test_ts = test_df.agg(F.max("timestamp")).collect()[0][0]
    total_weeks = int(np.ceil((max_test_ts - min_test_ts) / FIXED_TIME_INTERVAL))

    s2_weekly_history = []
    s1_weekly_history = []

    # =============================================================
    # MODEL 1: HYBRID MODEL WITH OPTIMAL 3D PARETO PARAMETERS
    # =============================================================
    print("\n [Model 1/2] Hybrid Model (Physical Retrain if avg(w2) >= tau*)...")
    s2_als = BaselineALS(rank=32, max_iter=max_iterations, reg_param=0.05)
    curr_train_s2 = cumulative_base_df
    s2_als.fit(curr_train_s2)

    current_week = 0
    s2_retrain_count = 0
    window_start = min_test_ts
    accumulated_weekly_dfs_s2 = []

    s2_overall_tracker = MetricTracker()
    t0_s2 = time.time()

    candidates, item_id_arr, item_matrix = update_lookup_matrices(s2_als)
    short_model = ShortTermModel(s2_als.item_factors, decay_gamma=gamma_star)

    while window_start < max_test_ts:
        window_end = window_start + FIXED_TIME_INTERVAL
        chunk_df = test_df.filter((F.col("timestamp") >= window_start) & (F.col("timestamp") < window_end))
        current_week += 1

        if chunk_df.count() > 0:
            chunk_stream = extract_stream_dict(chunk_df.sort("timestamp").collect())
            s2_week_tracker = MetricTracker()
            weekly_w2_list = []

            for u, interactions in chunk_stream.items():
                if u not in s2_als.user_factors:
                    continue

                seen_items = set(train_seen_map.get(u, []))
                session_buffer = []
                base_scores = s2_als.predict_user_scores(u, candidates)

                for i, (item_id, rating, ts) in enumerate(interactions):
                    gt_items = {item_id for item_id, r, _ in interactions[i:] if r >= 3.0}
                    if not gt_items:
                        seen_items.add(item_id)
                        continue

                    unseen_base_scores = {k: v for k, v in base_scores.items() if k not in seen_items}
                    top_k_base_masked = sorted(unseen_base_scores, key=unseen_base_scores.get, reverse=True)[:100]

                    if not session_buffer:
                        top_k_hybrid = top_k_base_masked
                        w1, w2 = 1.0, 0.0
                    else:
                        x_tilde = short_model.compute_centroid(session_buffer)
                        top_k_short, short_scores = get_fast_top_k_short(x_tilde, item_matrix, item_id_arr, k=100)
                        top_k_short_masked = [x for x in top_k_short if x not in seen_items]

                        w1, w2, _ = compute_hybrid_weights(len(session_buffer), top_k_base_masked, top_k_short_masked, lambda_param=lambda_star)
                        candidate_pool = (set(top_k_base_masked) | set(top_k_short_masked)) - seen_items
                        hybrid_scores = {
                            item: (w1 * base_scores.get(item, 0.0)) + (w2 * short_scores.get(item, 0.0))
                            for item in candidate_pool
                        }
                        top_k_hybrid = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:100]

                    weekly_w2_list.append(w2)

                    pred_rating = (w1 * base_scores.get(item_id, 3.5))
                    if session_buffer:
                        pred_rating += (w2 * short_scores.get(item_id, 0.0))

                    s2_week_tracker.add_sample(top_k_hybrid, gt_items, [(rating, pred_rating)])
                    s2_overall_tracker.add_sample(top_k_hybrid, gt_items, [(rating, pred_rating)])

                    session_buffer.append((item_id, rating))
                    seen_items.add(item_id)

            accumulated_weekly_dfs_s2.append(chunk_df)

            avg_weekly_w2 = float(np.mean(weekly_w2_list)) if weekly_w2_list else 0.0
            print(f" -> [Week {current_week}/{total_weeks}] Average w2: {avg_weekly_w2:.4f} | Threshold tau*: {tau_star:.2f}")

            week_metrics = s2_week_tracker.get_summary()
            is_retrained = (avg_weekly_w2 >= tau_star)
            s2_weekly_history.append({'week': current_week, 'metrics': week_metrics, 'retrained': is_retrained})

            if avg_weekly_w2 >= tau_star:
                s2_retrain_count += 1
                print_checkpoint_metrics("Hybrid Model (avg w2 >= tau*)", s2_retrain_count, current_week, total_weeks, week_metrics)

                combined_test_data = accumulated_weekly_dfs_s2[0]
                for w_df in accumulated_weekly_dfs_s2[1:]:
                    combined_test_data = combined_test_data.unionByName(w_df)

                curr_train_s2 = cumulative_base_df.unionByName(combined_test_data)
                s2_als.fit(curr_train_s2)

                candidates, item_id_arr, item_matrix = update_lookup_matrices(s2_als)
                short_model = ShortTermModel(s2_als.item_factors, decay_gamma=gamma_star)
            else:
                print(f"   -> [Hybrid Model Week {current_week}/{total_weeks}] avg(w2) < tau*. Physical retrain skipped.")

        window_start = window_end

    s2_time = time.time() - t0_s2

    # =============================================================
    # MODEL 2: BASELINE ALS (PHYSICAL RETRAIN EVERY 7 DAYS)
    # =============================================================
    print("\n [Model 2/2] Baseline ALS (Physical Retrain every 7 Days)...")
    s1_als = BaselineALS(rank=32, max_iter=max_iterations, reg_param=0.05)
    s1_als.fit(cumulative_base_df)

    current_week = 0
    s1_retrain_count = 0
    window_start = min_test_ts
    accumulated_weekly_dfs = []

    s1_overall_tracker = MetricTracker()
    t0_s1 = time.time()

    user_seen_history = {u: set(seen) for u, seen in train_seen_map.items()}

    while window_start < max_test_ts:
        window_end = window_start + FIXED_TIME_INTERVAL
        chunk_df = test_df.filter((F.col("timestamp") >= window_start) & (F.col("timestamp") < window_end))
        current_week += 1

        if chunk_df.count() > 0:
            chunk_stream = extract_stream_dict(chunk_df.sort("timestamp").collect())
            candidates = list(s1_als.item_factors.keys())
            s1_week_tracker = MetricTracker()

            for u, interactions in chunk_stream.items():
                if u not in s1_als.user_factors:
                    continue

                if u not in user_seen_history:
                    user_seen_history[u] = set()

                base_scores = s1_als.predict_user_scores(u, candidates)

                for i, (item_id, rating, ts) in enumerate(interactions):
                    gt_items = {item_id for item_id, r, _ in interactions[i:] if r >= 3.0}
                    if not gt_items:
                        user_seen_history[u].add(item_id)
                        continue

                    unseen_scores = {k: v for k, v in base_scores.items() if k not in user_seen_history[u]}
                    top_k_base = sorted(unseen_scores, key=unseen_scores.get, reverse=True)[:100]

                    raw_pred = base_scores.get(item_id, 3.5)
                    pred_rating = float(np.clip(raw_pred, 1.0, 5.0))

                    s1_week_tracker.add_sample(top_k_base, gt_items, [(rating, pred_rating)])
                    s1_overall_tracker.add_sample(top_k_base, gt_items, [(rating, pred_rating)])

                    user_seen_history[u].add(item_id)

            accumulated_weekly_dfs.append(chunk_df)
            combined_weekly_df = accumulated_weekly_dfs[0]
            for w_df in accumulated_weekly_dfs[1:]:
                combined_weekly_df = combined_weekly_df.unionByName(w_df)

            curr_train_s1 = cumulative_base_df.unionByName(combined_weekly_df)

            s1_retrain_count += 1
            week_metrics = s1_week_tracker.get_summary()
            s1_weekly_history.append({'week': current_week, 'metrics': week_metrics})

            print_checkpoint_metrics("Baseline ALS (7-Day)", s1_retrain_count, current_week, total_weeks, week_metrics)

            s1_als.fit(curr_train_s1)

        window_start = window_end

    s1_time = time.time() - t0_s1

    # -------------------------------------------------------------
    # OVERALL TEST SET EVALUATION SUMMARY TABLE
    # -------------------------------------------------------------
    m1_all = s1_overall_tracker.get_summary()
    m2_all = s2_overall_tracker.get_summary()

    summary_table_str = save_summary_table_file(m1_all, m2_all, s1_retrain_count, s2_retrain_count, s1_time, s2_time, best_params=best_params)
    print(summary_table_str)

    plot_weekly_test_performance(s1_weekly_history, s2_weekly_history)
    plot_overall_performance(m1_all, m2_all, s1_retrain_count, s2_retrain_count, s1_time, s2_time)
    print("\n -> All plots and evaluation summary successfully written to 'results_plots/' directory.\n")

    cumulative_base_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()