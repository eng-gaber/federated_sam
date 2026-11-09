import matplotlib.pyplot as plt
from data import load_results
import numpy as np
import torch
import os
import math

def get_centralized_results(filename):
    results = load_results(filename)

    eval_results = results["eval_results"]
    eval_loss = [m[0] for m in eval_results]
    eval_accuracy = [m[1].cpu().detach().item() for m in eval_results]
    eval_dsc = [m[2] for m in eval_results]
    eval_iou = [dsc / (2 - dsc) if dsc != 2 else 0 for dsc in eval_dsc]

    return {
        "loss_per_epoch":   results["loss_per_epoch"],
        "eval_loss":        eval_loss,
        "eval_accuracy":    eval_accuracy,
        "eval_dsc":         eval_dsc,
        "eval_iou":         eval_iou
    }

def get_federated_results(filename):
    results = load_results(filename)
    
    # Flower History object - access as attributes, not dict keys
    # Try to get loss from metrics_distributed_fit
    loss_per_epoch = []
    if hasattr(results, 'metrics_distributed_fit'):
        mdf = results.metrics_distributed_fit
        if isinstance(mdf, dict):
            if 'loss_per_epoch' in mdf:
                for round_idx, losses in mdf['loss_per_epoch']:
                    loss_per_epoch.extend(list(losses))
            elif 'losses' in mdf:
                for round_idx, losses in mdf['losses']:
                    loss_per_epoch.extend(list(losses))
        else:
            try:
                for round_idx, metrics_dict in mdf:
                    if 'losses' in metrics_dict:
                        loss_per_epoch.extend(metrics_dict['losses'])
            except Exception:
                pass
    
    # If no losses found, try losses_distributed
    if not loss_per_epoch and hasattr(results, 'losses_distributed'):
        loss_per_epoch = [m[1] for m in results.losses_distributed]

    eval_loss = [m[1] for m in results.losses_distributed]
    eval_accuracy = [m[1] for m in results.metrics_distributed["accuracy"]]
    eval_dsc = [m[1] for m in results.metrics_distributed["dsc"]]
    eval_iou = [dsc / (2 - dsc) if dsc != 2 else 0 for dsc in eval_dsc]

    return {
        "loss_per_epoch":   loss_per_epoch,
        "eval_loss":        eval_loss,
        "eval_accuracy":    eval_accuracy,
        "eval_dsc":         eval_dsc,
        "eval_iou":         eval_iou
    }

def plot_results(results, filename):
    names = ["loss_per_epoch", "eval_loss", "eval_accuracy", "eval_dsc", "eval_iou"]
    
    save_dir = os.path.join("plots", filename)
    os.makedirs(save_dir, exist_ok=True)
    
    for name in names:
        fig, ax = plt.subplots(figsize=(8, 6))
        y = results[name]
        x = np.arange(1, len(y)+1)

        max_y_ceil = math.ceil(max(y)) if len(y) > 0 else 0
        if max_y_ceil > 0:
            magnitude = 10**math.floor(math.log(max_y_ceil, 10))
            ax.set_yticks(np.arange(0, max_y_ceil+0.1*magnitude, 0.1*magnitude))
            ax.set_ylim(0, max_y_ceil+0.1*magnitude)
        else:
            ax.set_ylim(0, 1)
            
        ax.set_xlabel('Round')
        ax.set_ylabel(name.replace('_', ' ').title())
        
        if len(y) > 0:
            if 'loss' in name:
                best_val = min(y)
                best_text = f"Min Value: {best_val:.4f}"
            else:
                best_val = max(y)
                best_text = f"Max Value: {best_val:.4f}"
            ax.set_title(f"{filename}\n{name.replace('_', ' ').title()} | {best_text}")
        else:
            ax.set_title(f"{filename} - {name}")

        ax.plot(x, y)
        
        plt.savefig(os.path.join(save_dir, f"{name}.png"))
        plt.close(fig)

    print(f"Created separate plots for {filename} in {save_dir}/...")

if __name__ == "__main__":
    for f in os.listdir("results/"):
        filename = f[:-4] # remove .pkl extension
        if "centralized" in filename:
            results = get_centralized_results(filename)
        else:
            results = get_federated_results(filename)

        plot_results(results, filename)