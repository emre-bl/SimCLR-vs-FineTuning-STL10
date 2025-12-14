# final_evaluation.py
"""
Loads all three final models, evaluates them on the test set,
saves a text report and generates a comparison plot.
"""

import torch
import torch.nn as nn
from torchvision import models as torchvision_models
import matplotlib.pyplot as plt
import os
import config
import data_setup
import models
import utils

def save_plot(results, output_dir):
    """
    Generates and saves a bar chart comparing the models.
    """
    methods = list(results.keys())
    accuracies = [res['acc'] * 100 for res in results.values()]
    f1_scores = [res['f1'] * 100 for res in results.values()]

    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy (%)')
    rects2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='F1-Score (%)')

    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison (STL-10 Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 100)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Final Model Comparison on STL-10 Test Set ---")
    device = config.DEVICE
    criterion = nn.CrossEntropyLoss()
    
    # FIX: Initialize dictionary to store results by method name
    results = {}

    # --- 1. Evaluate Baseline Model ---
    print("\nEvaluating Baseline (from scratch) model...")
    try:
        baseline_model = models.get_baseline_model().to(device)
        baseline_model.load_state_dict(torch.load("baseline_model.pth"))
        
        test_transform_base = data_setup.get_baseline_transforms(train=False)
        test_loader_base = data_setup.get_stl10_loaders('test', test_transform_base, config.BATCH_SIZE_BASELINE, shuffle=False)
        
        base_loss, base_acc, base_f1 = utils.evaluate(baseline_model, test_loader_base, criterion, device)
        # FIX: Assign to specific key
        results['Baseline'] = {'acc': base_acc, 'f1': base_f1}
    except FileNotFoundError:
        print("Warning: baseline_model.pth not found. Skipping.")
        results['Baseline'] = {'acc': 0, 'f1': 0}

    # --- 2. Evaluate Fine-Tuned Model ---
    print("Evaluating Fine-Tuned (ImageNet) model...")
    try:
        finetune_model = models.get_finetune_model().to(device)
        finetune_model.load_state_dict(torch.load("finetune_model.pth"))
        
        test_transform_ft = data_setup.get_finetune_transforms()
        test_loader_ft = data_setup.get_stl10_loaders('test', test_transform_ft, config.BATCH_SIZE_FINETUNE, shuffle=False)
        
        ft_loss, ft_acc, ft_f1 = utils.evaluate(finetune_model, test_loader_ft, criterion, device)
        # FIX: Assign to specific key
        results['Fine-Tuned'] = {'acc': ft_acc, 'f1': ft_f1}
    except FileNotFoundError:
        print("Warning: finetune_model.pth not found. Skipping.")
        results['Fine-Tuned'] = {'acc': 0, 'f1': 0}

    # --- 3. Evaluate Self-Supervised (SimCLR) Model ---
    print("Evaluating Self-Supervised (SimCLR) model...")
    try:
        backbone = torchvision_models.resnet50(weights=None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
        ssl_model = nn.Sequential(
            backbone,
            nn.Linear(num_ftrs, config.NUM_CLASSES)
        ).to(device)
        
        ssl_model.load_state_dict(torch.load("linear_eval_model.pth"))
        
        test_transform_base = data_setup.get_baseline_transforms(train=False) 
        test_loader_ssl = data_setup.get_stl10_loaders('test', test_transform_base, config.BATCH_SIZE_BASELINE, shuffle=False)

        ssl_loss, ssl_acc, ssl_f1 = utils.evaluate(ssl_model, test_loader_ssl, criterion, device)
        # FIX: Assign to specific key
        results['Self-Supervised'] = {'acc': ssl_acc, 'f1': ssl_f1}
    except FileNotFoundError:
        print("Warning: linear_eval_model.pth not found. Skipping.")
        results['Self-Supervised'] = {'acc': 0, 'f1': 0}

    # --- 4. Save Report and Plot ---
    report_path = os.path.join(output_dir, 'final_report.txt')
    with open(report_path, 'w') as f:
        header = f"{'Method':<25} | {'Test Accuracy':<15} | {'Test F1-Score':<15}"
        sep = "-" * len(header)
        
        print("\n" + "="*60)
        print("     Final Comparative Analysis Results")
        print("="*60)
        print(header)
        print(sep)
        
        f.write("CMP722 - Final Comparative Analysis Results\n")
        f.write("="*60 + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")

        for method, metrics in results.items():
            line = f"{method:<25} | {metrics['acc']:<15.4f} | {metrics['f1']:<15.4f}"
            print(line)
            f.write(line + "\n")
        
        print("="*60)
        f.write("="*60 + "\n")
    
    print(f"\nFinal report saved to {report_path}")
    
    try:
        save_plot(results, output_dir)
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()