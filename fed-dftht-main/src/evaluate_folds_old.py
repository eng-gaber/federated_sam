import torch
from models import get_model
from data import load_polypgen_centralized, load_polypgen_federated, load_config, get_polyp_size, check_small_polyp
import torchmetrics
import os
import wandb
import numpy as np
from collections import defaultdict
from skimage import io
from skimage.metrics import hausdorff_distance
import albumentations as A
import pandas as pd
import re

from scipy.stats import t
from sklearn.metrics import average_precision_score

CKPT_DIR = os.path.join("ckpts", "checkpoints_owner_SDUTNET_FEDAVG")

model_names =   [
                    "sd_utnet", "sdnet", "utnet"
                ]

def discover_checkpoint_sets(ckpt_dir):
    pattern = re.compile(r'^(?P<ckpt_name>.+)_fold(?P<fold>\d+)_(?P<suffix>.+)\.pt$')
    ckpt_sets = {}
    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if not match:
            continue
        ckpt_name = match.group('ckpt_name')
        fold = int(match.group('fold'))
        suffix = match.group('suffix')
        ckpt_sets.setdefault(ckpt_name, {}).setdefault(fold, {})[suffix] = os.path.join(ckpt_dir, fname)

    checkpoint_sets = {}
    for ckpt_name, folds in ckpt_sets.items():
        checkpoint_sets[ckpt_name] = {}
        for fold, variants in folds.items():
            best_path = None
            best_small_path = None
            for suffix, path in variants.items():
                if suffix.endswith('best_small'):
                    best_small_path = path
                elif suffix.endswith('best'):
                    best_path = path

            if best_path is None or best_small_path is None:
                raise RuntimeError(f"Missing best/best_small checkpoint for {ckpt_name} fold {fold} in {ckpt_dir}")

            checkpoint_sets[ckpt_name][fold] = {
                'best': best_path,
                'best_small': best_small_path
            }

    return checkpoint_sets


def load_checkpoint_state_dict(ckpt_path, device):
    """Load checkpoint and return state_dict, handling potential wrappers."""
    if os.name == 'nt':
        abs_path = os.path.abspath(ckpt_path)
        if not abs_path.startswith('\\\\?\\'):
            ckpt_path = '\\\\?\\' + abs_path
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'net', 'model'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return ckpt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_sets = discover_checkpoint_sets(CKPT_DIR)
    if not checkpoint_sets:
        raise RuntimeError(f"No checkpoint files found in {CKPT_DIR}.")

    wandb.init(project="test_folds",
               config={'ckpt_dir': CKPT_DIR, 'ckpt_sets': list(checkpoint_sets.keys())})
    
    metric_names = ["dice_s", "dice_t", "dice_l", "dice_n", "dice"]

    columns = ["ckpt_name"] + metric_names
    print(columns)
    table_val = wandb.Table(columns=columns)
    table_in = wandb.Table(columns=columns)
    table_out = wandb.Table(columns=columns)

    dataset = "polypgen"

    dict_keys = ["in_test_" + metric for metric in metric_names]
    dict_keys.extend(["out_test_" + metric for metric in metric_names])
    df_dict = {dict_key: dict() for dict_key in dict_keys}

    best_score_dict = {dict_key: 0 for dict_key in dict_keys}
    best_name_dict = {dict_key: None for dict_key in dict_keys}
    ckpt_type_names = []

    print(f"Discovered checkpoint sets: {list(checkpoint_sets.keys())}")

    # To store ratios across folds
    ratios_by_fold = {}

    for ckpt_name, ckpt_folds in checkpoint_sets.items():
        for name in model_names:
            if name in ckpt_name:
                model_name = name
                break
        else:
            raise RuntimeError(f"Unable to infer model type from checkpoint name '{ckpt_name}'")

        net = get_model(model_name, in_channels=3, use_lightning=False, local=False)
        print(ckpt_name, model_name)

        fold_numbers = sorted(ckpt_folds.keys())
        best_fold_dict = {fold_i: {} for fold_i in fold_numbers}
        small_fold_dict = {fold_i: {} for fold_i in fold_numbers}

        for fold_i in fold_numbers:
            best_ckpt_path = ckpt_folds[fold_i]['best']
            small_ckpt_path = ckpt_folds[fold_i]['best_small']
            
            best_ckpt = load_checkpoint_state_dict(best_ckpt_path, device)
            small_ckpt = load_checkpoint_state_dict(small_ckpt_path, device)

            dataloaders, num_examples = load_polypgen_federated(None, data_config={
                                                "batch_size": 4,
                                                "target_size": 512,
                                                "out_center": 6,
                                                "splits": "per_fold",
                                                "seed": 1,
                                                "load_in_ram": False,
                                                "load_names": False,
                                                "fold_nr": fold_i})
            
            for ckpt, dict_ckpt in [(best_ckpt, best_fold_dict), (small_ckpt, small_fold_dict)]:
                net.load_state_dict(ckpt)
                net = net.to(device)

                dict_ckpt[fold_i]['val'], dict_ckpt[fold_i]['val_scores'], val_ratio = test(net, dataloaders['val'], device, dataset)
                dict_ckpt[fold_i]['in'], dict_ckpt[fold_i]['in_scores'], in_ratio = test(net, dataloaders['in_test'], device, dataset)
                dict_ckpt[fold_i]['out'], dict_ckpt[fold_i]['out_scores'], out_ratio = test(net, dataloaders['out_test'], device, dataset)
                
                # Keep track of ratios
                ratios_by_fold[fold_i] = {
                    'val': val_ratio,
                    'in': in_ratio,
                    'out': out_ratio
                }
        
        ckpt_i = 0
        # Average over all folds
        for ckpt_type, dict_ckpt in [("best", best_fold_dict), ("best_small", small_fold_dict)]:
            ckpt_type_name = ckpt_name + "_" + ckpt_type
            ckpt_type_names.append(ckpt_type_name)
            n_col = len(metric_names)
            n_folds = len(fold_numbers)
            val_total = np.zeros((n_folds, n_col))
            in_total = np.zeros((n_folds, n_col))
            out_total = np.zeros((n_folds, n_col))

            for i, fold_i in enumerate(fold_numbers):
                val_total[i] = dict_ckpt[fold_i]['val']
                in_total[i] = dict_ckpt[fold_i]['in']
                out_total[i] = dict_ckpt[fold_i]['out']

            for i, metric in enumerate(metric_names):
                in_m = "in_test_" + metric
                out_m = "out_test_" + metric
                df_dict[in_m][ckpt_type_name] = np.array([dict_ckpt[fold_i]['in'][i] for fold_i in fold_numbers])
                df_dict[out_m][ckpt_type_name] = np.array([dict_ckpt[fold_i]['out'][i] for fold_i in fold_numbers])

            res_val = (np.mean(val_total, axis=0)).tolist()
            res_val = [f"{res_val[i]:.4f}, std. {np.std(val_total[:, i]):.4f}" for i in range(len(res_val))]

            res_in = (np.mean(in_total, axis=0)).tolist()
            res_in_str = [f"{res_in[i]:.4f}, std. {np.std(in_total[:, i]):.4f}" for i in range(len(res_in))]

            res_out = (np.mean(out_total, axis=0)).tolist()
            res_out_str = [f"{res_out[i]:.4f}, std. {np.std(out_total[:, i]):.4f}" for i in range(len(res_out))]

            table_val.add_data(ckpt_type_name, *res_val)
            table_in.add_data(ckpt_type_name, *res_in_str)
            table_out.add_data(ckpt_type_name, *res_out_str)

            for i, metric in enumerate(metric_names):
                in_test_m = "in_test_" + metric
                out_test_m = "out_test_" + metric

                if res_in[i] > best_score_dict[in_test_m]:
                    best_score_dict[in_test_m] = res_in[i]
                    best_name_dict[in_test_m] = ckpt_type_name

                if res_out[i] > best_score_dict[out_test_m]:
                    best_score_dict[out_test_m] = res_out[i]
                    best_name_dict[out_test_m] =  ckpt_type_name

            ckpt_i += 1

            # Save individual scores and per-fold summary to csv
            save_individual_scores(ckpt_type_name, dict_ckpt, fold_numbers)
            save_fold_summaries(ckpt_type_name, dict_ckpt, fold_numbers)

    wandb.log({f"Table validation": table_val})
    wandb.log({f"Table in_test": table_in})
    wandb.log({f"Table out_test": table_out})

    table_ratios = wandb.Table(columns=['fold', 'val', 'in-test', 'out-test'])
    for fold_i in sorted(ratios_by_fold.keys()):
        table_ratios.add_data(
            fold_i, 
            ratios_by_fold[fold_i]['val'], 
            ratios_by_fold[fold_i]['in'], 
            ratios_by_fold[fold_i]['out']
        )

    wandb.log({f"Table ratios": table_ratios})
    print()


def save_individual_scores(ckpt_type_name, dict_ckpt, fold_numbers):
    for scores_set in ["val_scores", "in_scores", "out_scores"]:
        df_rows = []
        for fold_i in fold_numbers:
            fold_scores = dict_ckpt[fold_i][scores_set]
            for score in fold_scores:
                df_rows.append([
                    fold_i,
                    score[0],
                    score[1],
                    score[2],
                    score[3],
                    score[4],
                    score[5],
                    score[6],
                    score[7],
                    score[8],
                    score[9],
                    score[10]
                ])
        df = pd.DataFrame(df_rows, columns=[
            "fold_nr", "img_path", "dsc", "iou", "recall", "precision",
            "f2", "jaccard", "hd", "ap", "small_polyp", "inv_area"
        ])
        os.makedirs("fold_results_fedAVG", exist_ok=True)

        df.to_csv(
            os.path.join("fold_results_fedAVG", f"{ckpt_type_name}_{scores_set}.csv"),
            index=False
        )
    


def save_fold_summaries(ckpt_type_name, dict_ckpt, fold_numbers):
    os.makedirs("fold_results_fedAVG", exist_ok=True)
    for fold_i in fold_numbers:
        summary_rows = []
        for split in ["val", "in", "out"]:
            scores = dict_ckpt[fold_i][f"{split}_scores"]
            if not scores:
                continue

            dice_list = [s[1] for s in scores if not np.isnan(s[1])]
            iou_list = [s[2] for s in scores if not np.isnan(s[2])]
            recall_list = [s[3] for s in scores if not np.isnan(s[3])]
            precision_list = [s[4] for s in scores if not np.isnan(s[4])]
            f2_list = [s[5] for s in scores if not np.isnan(s[5])]
            jacc_list = [s[6] for s in scores if not np.isnan(s[6])]
            hd_list = [s[7] for s in scores if not np.isnan(s[7])]
            ap_list = [s[8] for s in scores if not np.isnan(s[8])]

            summary_rows.append([
                fold_i,
                split,
                len(scores),
                np.mean(dice_list) if dice_list else 0.0,
                np.mean(iou_list) if iou_list else 0.0,
                np.mean(recall_list) if recall_list else 0.0,
                np.mean(precision_list) if precision_list else 0.0,
                np.mean(f2_list) if f2_list else 0.0,
                np.mean(jacc_list) if jacc_list else 0.0,
                np.mean(hd_list) if hd_list else 0.0,
                np.mean(ap_list) if ap_list else 0.0
            ])

        df = pd.DataFrame(summary_rows, columns=[
            "fold_nr", "split", "n_images",
            "mean_dsc", "mean_iou", "mean_recall", "mean_precision",
            "mean_f2", "mean_jaccard", "mean_hd", "mean_ap"
        ])
        df.to_csv(os.path.join("fold_results_fedAVG", f"{ckpt_type_name}_fold{fold_i}_summary.csv"), index=False)


def test(net, dataloader, device, dataset):
    net.eval()
    threshold = 150 if dataset == "polypgen" else 1000
    dice_scores = defaultdict(lambda: list())
    img_scores = []

    with torch.no_grad():
        for images, y_true, img_path, _ in dataloader:
            images, y_true = images.to(device), y_true.to(device)

            out = net(images)
            out = out if type(out) is tuple else tuple([out])
            
            # Convert network output to predicted class mask
            pred = torch.argmax(out[0], dim=1)

            # Binary foreground masks (polyp = class 1)
            pred = (pred == 1).float()
            target = (y_true == 1).float()

            prob = torch.softmax(out[0], dim=1)[:, 1, ...]

            # Iterate through individual images in the batch to calculate per-sample metrics
            batch_size_actual = images.size(0)
            for b in range(batch_size_actual):
                single_img_path = img_path[b]
                small_polyp, inv_area = check_small_polyp(single_img_path, True, dataset)

                single_pred = pred[b]
                single_target = target[b]
                single_prob = prob[b]

                intersection = (single_pred * single_target).sum()
                union = ((single_pred + single_target) > 0).float().sum()

                dsc = (
                    (2.0 * intersection + 1e-7)
                    / (single_pred.sum() + single_target.sum() + 1e-7)
                ).item()
                iou = ((intersection + 1e-7) / (union + 1e-7)).item()

                true_positives = (single_pred * single_target).sum().item()
                false_positives = (single_pred * (1 - single_target)).sum().item()
                false_negatives = ((1 - single_pred) * single_target).sum().item()

                precision = true_positives / (true_positives + false_positives + 1e-7)
                recall = true_positives / (true_positives + false_negatives + 1e-7)
                f2 = (5 * precision * recall) / (4 * precision + recall + 1e-7)
                jaccard = iou

                if single_target.sum() > 0 and single_pred.sum() > 0:
                    try:
                        # Squeeze or index array for correct 2D Hausdorff Distance calculation
                        hd = hausdorff_distance(
                            single_target.cpu().numpy().astype(bool), 
                            single_pred.cpu().numpy().astype(bool)
                        )
                    except Exception:
                        hd = float('nan')
                else:
                    hd = float('nan')

                dice_scores["dice"].append(dsc)

                if inv_area > 0:
                    dice_scores["dice_n"].append(dsc)

                    if small_polyp:
                        dice_scores["dice_s"].append(dsc)
                        if inv_area >= threshold:
                            dice_scores["dice_t"].append(dsc)
                    else:
                        dice_scores["dice_l"].append(dsc)

                # Compute Average Precision for this single image
                if single_target.sum() > 0:
                    ap = average_precision_score(
                        single_target.detach().cpu().numpy().ravel(), 
                        single_prob.detach().cpu().numpy().ravel()
                    )
                else:
                    ap = float('nan')

                img_scores.append([
                    single_img_path,
                    dsc,
                    iou,
                    recall,
                    precision,
                    f2,
                    jaccard,
                    hd,
                    ap,
                    small_polyp,
                    inv_area
                ])

    dice_s = np.mean(dice_scores['dice_s']) if dice_scores['dice_s'] else 0.0
    dice_t = np.mean(dice_scores['dice_t']) if dice_scores['dice_t'] else 0.0
    dice_l = np.mean(dice_scores['dice_l']) if dice_scores['dice_l'] else 0.0
    dice_n = np.mean(dice_scores['dice_n']) if dice_scores['dice_n'] else 0.0
    dice = np.mean(dice_scores['dice']) if dice_scores['dice'] else 0.0

    arr = np.array([dice_s, dice_t, dice_l, dice_n, dice])
    
    # Helper to prevent division-by-zero crashes
    def get_len(lst):
        return len(lst)
        
    len_s = get_len(dice_scores['dice_s'])
    len_t = get_len(dice_scores['dice_t'])
    len_l = get_len(dice_scores['dice_l'])
    len_n = get_len(dice_scores['dice_n'])
    len_all = get_len(dice_scores['dice'])

    ratio_s_n = (len_s / len_n) if len_n > 0 else 0.0
    ratio_t_s = (len_t / len_s) if len_s > 0 else 0.0
    ratio_l_n = (len_l / len_n) if len_n > 0 else 0.0
    ratio_n_all = (len_n / len_all) if len_all > 0 else 0.0

    ratios = (f"dice_s: {ratio_s_n:.4f} | "
              f"dice_t: {ratio_t_s:.4f} | "
              f"dice_l: {ratio_l_n:.4f} | "
              f"dice_n: {ratio_n_all:.4f}")

    return arr, img_scores, ratios

if __name__ == "__main__":
    main()
