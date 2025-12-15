import torch
import torch.nn as nn
from torchvision import models as torchvision_models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import os
import config
import data_setup
import models
import utils

# STL-10 Class Names
CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def load_model_weights(model, filename, output_dir, device):
    """
    Attempts to load model weights from output_dir first, then current directory.
    """
    # 1. Check results directory
    path_in_results = os.path.join(output_dir, filename)
    if os.path.exists(path_in_results):
        print(f"Loading {filename} from results directory...")
        model.load_state_dict(torch.load(path_in_results, map_location=device))
        return True
    
    # 2. Check root directory (Fallback)
    if os.path.exists(filename):
        print(f"Loading {filename} from root directory...")
        model.load_state_dict(torch.load(filename, map_location=device))
        return True
        
    print(f"Warning: {filename} not found in {output_dir} or root.")
    return False

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion Matrix saved to {filename}")

def save_comparison_plot(results, output_dir):
    """
    Generates a grouped bar chart for Accuracy, F1, Precision, and Recall.
    """
    methods = list(results.keys())
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    # Extract data
    data = {m: [] for m in metrics}
    for method in methods:
        data['Accuracy'].append(results[method]['acc'] * 100)
        data['F1-Score'].append(results[method]['f1'] * 100)
        data['Precision'].append(results[method]['prec'] * 100)
        data['Recall'].append(results[method]['rec'] * 100)

    x = np.arange(len(methods))
    width = 0.2  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    
    rects1 = ax.bar(x - 1.5*width, data['Accuracy'], width, label='Accuracy')
    rects2 = ax.bar(x - 0.5*width, data['F1-Score'], width, label='F1-Score')
    rects3 = ax.bar(x + 0.5*width, data['Precision'], width, label='Precision')
    rects4 = ax.bar(x + 1.5*width, data['Recall'], width, label='Recall')

    ax.set_ylabel('Score (%)')
    ax.set_title('Comprehensive Model Comparison (STL-10)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'final_comparison_plot.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

def evaluate_model_full(model, loader, device, name, output_dir):
    """
    Runs full evaluation returning all metrics and generating CM.
    """
    print(f"Evaluating {name}...")
    y_true, y_pred = utils.get_all_predictions(model, loader, device)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Generate Confusion Matrix
    cm_filename = os.path.join(output_dir, f'cm_{name.lower().replace(" ", "_")}.png')
    plot_confusion_matrix(y_true, y_pred, f'Confusion Matrix - {name}', cm_filename)
    
    return {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec}

def main():
    output_dir = getattr(config, 'RESULTS_DIR', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Final Model Comparison with Advanced Metrics ---")
    device = config.DEVICE
    results = {}

    # --- 1. Baseline ---
    baseline_model = models.get_baseline_model().to(device)
    if load_model_weights(baseline_model, "baseline_model.pth", output_dir, device):
        test_loader = data_setup.get_stl10_loaders('test', data_setup.get_baseline_transforms(train=False), config.BATCH_SIZE_BASELINE, shuffle=False)
        results['Baseline'] = evaluate_model_full(baseline_model, test_loader, device, "Baseline", output_dir)
    else:
        results['Baseline'] = {'acc': 0, 'f1': 0, 'prec': 0, 'rec': 0}

    # --- 2. Fine-Tuned ---
    finetune_model = models.get_finetune_model().to(device)
    if load_model_weights(finetune_model, "finetune_model.pth", output_dir, device):
        test_loader_ft = data_setup.get_stl10_loaders('test', data_setup.get_finetune_transforms(), config.BATCH_SIZE_FINETUNE, shuffle=False)
        results['Fine-Tuned'] = evaluate_model_full(finetune_model, test_loader_ft, device, "Fine-Tuned", output_dir)
    else:
        results['Fine-Tuned'] = {'acc': 0, 'f1': 0, 'prec': 0, 'rec': 0}

    # --- 3. Self-Supervised (SimCLR) ---
    backbone = torchvision_models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    ssl_model = nn.Sequential(backbone, nn.Linear(2048, config.NUM_CLASSES)).to(device)
    
    if load_model_weights(ssl_model, "linear_eval_model.pth", output_dir, device):
        # Reuse baseline test loader
        test_loader_ssl = data_setup.get_stl10_loaders('test', data_setup.get_baseline_transforms(train=False), config.BATCH_SIZE_LINEAR, shuffle=False)
        results['SimCLR'] = evaluate_model_full(ssl_model, test_loader_ssl, device, "SimCLR", output_dir)
    else:
        results['SimCLR'] = {'acc': 0, 'f1': 0, 'prec': 0, 'rec': 0}

    # --- 4. Save Text Report ---
    report_path = os.path.join(output_dir, 'final_detailed_report.txt')
    with open(report_path, 'w') as f:
        header = f"{'Method':<20} | {'Acc':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}"
        sep = "-" * len(header)
        
        print("\n" + "="*65)
        print("       FINAL COMPARATIVE RESULTS (Detailed)")
        print("="*65)
        print(header)
        print(sep)
        
        f.write("CMP722 - Final Detailed Results\n")
        f.write("="*65 + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")

        for method, m in results.items():
            line = f"{method:<20} | {m['acc']:.4f}   | {m['f1']:.4f}   | {m['prec']:.4f}   | {m['rec']:.4f}"
            print(line)
            f.write(line + "\n")
        
        print("="*65)

    # --- 5. Generate Comparison Plot ---
    try:
        save_comparison_plot(results, output_dir)
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()