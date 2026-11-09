import flwr as fl
import hydra
import torch
import pandas as pd
import os
import shutil
import random
import numpy as np

from omegaconf import OmegaConf
from server import get_strategy
from client import get_client_fn
from data import save_results, load_polypgen_federated, load_polypgen_federated_dftht, load_model_checkpoint, delete_centers_textfile, POLYP_CENTERS, CKPTS_ROOT
from data_lits import load_lits_federated, LITS_CENTERS
from utils import extract_data_config
from models import get_model
# from lightning.pytorch.loggers import WandbLogger
from OLD_evaluate_ckpt import test
from scipy.spatial.distance import directed_hausdorff

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def compute_detailed_metrics(y_pred_bin, y_true):
    # Convert inputs to binary numpy arrays (0 or 1)
    if isinstance(y_pred_bin, torch.Tensor):
        y_pred_bin = y_pred_bin.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        
    y_pred_bin = (y_pred_bin > 0.5).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)
    
    y_pred_f = y_pred_bin.flatten()
    y_true_f = y_true.flatten()
    
    intersection = np.sum(y_pred_f * y_true_f)
    sum_pred = np.sum(y_pred_f)
    sum_true = np.sum(y_true_f)
    
    # DSC (Dice)
    dsc = (2.0 * intersection) / (sum_pred + sum_true) if (sum_pred + sum_true) > 0 else 1.0
    
    # IoU (Jaccard)
    union = sum_pred + sum_true - intersection
    iou = intersection / union if union > 0 else 1.0
    
    # Precision & Recall
    precision = intersection / sum_pred if sum_pred > 0 else (1.0 if sum_true == 0 else 0.0)
    recall = intersection / sum_true if sum_true > 0 else (1.0 if sum_pred == 0 else 0.0)
    
    # F2-score: (5 * Precision * Recall) / (4 * Precision + Recall)
    if (4 * precision + recall) > 0:
        f2 = (5.0 * precision * recall) / (4.0 * precision + recall)
    else:
        f2 = 1.0 if (sum_pred + sum_true == 0) else 0.0
        
    # Hausdorff Distance (HD)
    coords_pred = np.argwhere(y_pred_bin > 0)
    coords_true = np.argwhere(y_true > 0)
    if len(coords_pred) == 0 and len(coords_true) == 0:
        hd = 0.0
    elif len(coords_pred) == 0 or len(coords_true) == 0:
        hd = np.sqrt(y_pred_bin.shape[0]**2 + y_pred_bin.shape[1]**2)
    else:
        d1 = directed_hausdorff(coords_pred, coords_true)[0]
        d2 = directed_hausdorff(coords_true, coords_pred)[0]
        hd = max(d1, d2)
        
    return {
        "iou": iou,
        "dsc": dsc,
        "recall": recall,
        "precision": precision,
        "f2": f2,
        "hd": hd
    }

def evaluate_dataloader_detailed(net, dataloader, device):
    net.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, y_true, _, _ = batch
            else:
                images, y_true = batch
                
            images = images.to(device)
            out = net(images)
            out = out if type(out) is tuple else tuple([out])
            
            y_pred = out[0][:, 1, :, :].unsqueeze(dim=1)
            
            for i in range(images.shape[0]):
                m = compute_detailed_metrics(y_pred[i], y_true[i])
                all_metrics.append(m)
                
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    else:
        avg_metrics = {"iou": 0.0, "dsc": 0.0, "recall": 0.0, "precision": 0.0, "f2": 0.0, "hd": 0.0}
        
    return avg_metrics

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    data_config = extract_data_config(config)
    
    # Fix seed
    os.environ["PL_GLOBAL_SEED"] = str(config.data.seed)
    random.seed(config.data.seed)
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed(config.data.seed)

    # Fixes Ray-related errors on Snellius
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "num_cpus": 8,
        "num_gpus": 1,
        # "_temp_dir": "/home/pschutte/tmp"
    }

    print(f"Using the following configuration ({config.name})...\n{OmegaConf.to_yaml(config)}")

    load_dataset = load_polypgen_federated if config.data.dataset == "polypgen" else load_lits_federated
    centers = POLYP_CENTERS if config.data.dataset == "polypgen" else LITS_CENTERS

    client_centers = [c for c in centers if c != config.data.out_center]

    # Compute total_train_size in a lightweight manner (without constructing full data_collection/amp pools)
    total_train_size = 0
    for c in client_centers:
        _, num_ex = load_dataset(c, data_config)
        total_train_size += num_ex['train']

    # Server-side loader always uses the standard (non-augmented) pipeline
    server_dataloaders, num_examples = load_dataset(None, data_config)
    print(f"Succesfully created all dataloaders...\nServer num_examples: {num_examples}")

    wandb_logger = None

    # Directory with saved style parameters of clients in between rounds
    if os.path.isdir(os.path.join(CKPTS_ROOT, f"temp_{config.name}")):
        shutil.rmtree(os.path.join(CKPTS_ROOT, f"temp_{config.name}"))
    os.mkdir(os.path.join(CKPTS_ROOT, f"temp_{config.name}"))

    num_rounds = config.server.num_rounds
    if config.server.resume_cfg_name is not None and config.server.resume_from_round is not None:
        num_rounds = config.server.num_rounds - config.server.resume_from_round
        print(f"Resuming simulation from round {config.server.resume_from_round}. Running the remaining {num_rounds} rounds.")

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(config, total_train_size, None),
        num_clients=config.server.num_clients,
        client_resources={'num_cpus': config.client.num_cpus,
                          'num_gpus': config.client.num_gpus},
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(config, server_dataloaders['val'], wandb_logger),
        ray_init_args=ray_init_args
    )

    save_results(history, config)

    delete_centers_textfile(config.name)

    ### TESTING ###

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = get_model(config, use_lightning=False, local=False).to(device)

    df = pd.DataFrame({
        "in/out-test": [], "ckpt": [], "loss": [], "accuracy": [], "1-loss": [], "dsc": []
    })

    for suffix in ["best", "last"]:
        print(f"{suffix} checkpoint results:")
        load_model_checkpoint(net, f"federated_{config.name}", suffix)
        res = test(net, server_dataloaders['in_test'], device)
        print("[in_test]", res)
        df.loc[len(df)] = {"in/out-test": "in-test", "ckpt": suffix, "loss": res[0], "accuracy": res[1].cpu(), "1-loss": res[2], "dsc": res[3].cpu()}

        res = test(net, server_dataloaders['out_test'], device)
        print("[out_test]", res)
        print()
        df.loc[len(df)] = {"in/out-test": "out-test", "ckpt": suffix, "loss": res[0], "accuracy": res[1].cpu(), "1-loss": res[2], "dsc": res[3].cpu()}

    if wandb_logger is not None:
        wandb_logger.log_table(key="test_metrics", dataframe=df)

    # Detailed final evaluation for paper research
    print("Computing detailed metrics for paper research...")
    detailed_results = []
    client_centers = [c for c in centers if c != config.data.out_center]
    
    for suffix in ["best", "last"]:
        load_model_checkpoint(net, f"federated_{config.name}", suffix)
        
        # Evaluate each participating client's validation dataset
        for center_nr in client_centers:
            client_dataloaders, _ = load_dataset(center_nr, data_config)
            client_val_loader = client_dataloaders['val']
            metrics = evaluate_dataloader_detailed(net, client_val_loader, device)
            row = {
                "checkpoint": suffix,
                "dataset_name": f"client_{center_nr}_val",
                **metrics
            }
            detailed_results.append(row)
            print(f"[Client {center_nr} Val] - {suffix} checkpoint metrics: {metrics}")
            
        # Evaluate combined in_test
        in_metrics = evaluate_dataloader_detailed(net, server_dataloaders['in_test'], device)
        detailed_results.append({
            "checkpoint": suffix,
            "dataset_name": "in_test",
            **in_metrics
        })
        print(f"[In Test] - {suffix} checkpoint metrics: {in_metrics}")
        
        # Evaluate combined out_test
        out_metrics = evaluate_dataloader_detailed(net, server_dataloaders['out_test'], device)
        detailed_results.append({
            "checkpoint": suffix,
            "dataset_name": "out_test",
            **out_metrics
        })
        print(f"[Out Test] - {suffix} checkpoint metrics: {out_metrics}")
        
    detailed_df = pd.DataFrame(detailed_results)
    results_csv_path = os.path.join("results", f"{config.name}_detailed_metrics.csv")
    os.makedirs("results", exist_ok=True)
    detailed_df.to_csv(results_csv_path, index=False)
    print(f"Saved detailed metrics for paper research to {results_csv_path}")

    shutil.rmtree(os.path.join(CKPTS_ROOT, f"temp_{config.name}"))

if __name__ == "__main__":
    main()