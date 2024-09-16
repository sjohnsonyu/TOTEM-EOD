import os
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn


def run_baselines_if_not_exists(test_data, args):
    metrics_base_dir = 'imputation/metrics/'
    dataset_name = args.trained_vqvae_model_path.split('/')[2]
    upper_subdir = args.trained_vqvae_model_path.split('/')[3]
    metrics_dir = os.path.join(metrics_base_dir, dataset_name, upper_subdir)
    os.makedirs(metrics_dir, exist_ok=True)

    baseline_metrics_filename = os.path.join(metrics_dir, 'all_zeros_classification_metrics.csv')
    if not os.path.exists(baseline_metrics_filename):
        print('Baseline metrics not found. Running baselines...')
        labels = test_data[:, -1]
        run_baselines(labels, metrics_dir)
    else:
        print('Baseline metrics already exist. Skipping baseline evaluation.')


def get_sklearn_report_and_save(preds, trues, baseline_name, metrics_dir):
    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    trues_tensor = torch.tensor(trues, dtype=torch.float32)
    criterion = nn.BCELoss(reduction='none')
    nll = criterion(preds_tensor, trues_tensor)
    avg_negative_log_likelihood = nll.mean().item()
    perplexity = np.exp(avg_negative_log_likelihood)

    metrics = sklearn.metrics.precision_recall_fscore_support(trues, preds, average=None)
    metrics_df = pd.DataFrame({
        'Class': [0, 1],
        'Precision': metrics[0],
        'Recall': metrics[1],
        'F1-Score': metrics[2],
        'Support': metrics[3]
    })
    print(metrics_df.round(3))
    f1 = metrics_df['F1-Score'].mean()
    print(f'Overall F1-Score for {baseline_name}:', f1.round(3))

    auc_roc = sklearn.metrics.roc_auc_score(trues, preds)
    print(f'AUC-ROC for {baseline_name}:', round(auc_roc, 3))

    metrics_df.to_csv(os.path.join(metrics_dir, f'{baseline_name}_classification_metrics.csv'), index=False)

    overall_metrics = {
        'Overall F1-Score': f1,
        'AUC-ROC': auc_roc,
        'Perplexity': perplexity
    }
    overall_metrics_df = pd.DataFrame(list(overall_metrics.items()), columns=['Metric', 'Value'])
    overall_metrics_df.to_csv(os.path.join(metrics_dir, f'{baseline_name}_overall_metrics.csv'), index=False)


def run_baselines(test_data, metrics_dir):
    test_data = test_data.flatten()
    all_0_preds = np.zeros_like(test_data)
    all_1_preds = np.ones_like(test_data)

    p_1 = np.sum(test_data) / test_data.size
    random_preds = np.random.choice([0, 1], size=test_data.size, p=[1 - p_1, p_1])

    print('All 0s')
    get_sklearn_report_and_save(all_0_preds, test_data, 'all_zeros', metrics_dir)
    print('All 1s')
    get_sklearn_report_and_save(all_1_preds, test_data, 'all_ones', metrics_dir)
    print('Random')
    get_sklearn_report_and_save(random_preds, test_data, 'random', metrics_dir)

    print('-------------')