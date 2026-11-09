import argparse
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

DEFAULT_CKPT_DIR = os.path.join("ckpts", "final_checkpoints_SDUTNET_DFTHT_composite")
DEFAULT_CENTERS = [1, 2, 3, 4, 5]
DEFAULT_FOLDS = [1, 2, 3, 4, 5]

model_names = [
    "sd_utnet", "sdnet", "utnet"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fold checkpoints on per-center, per-fold PolypGen in_test datasets."
    )
    parser.add_argument(
        "--ckpt_dir",
        default=DEFAULT_CKPT_DIR,
        help="Checkpoint directory containing best fold models."
    )
    parser.add_argument(
        "--centers",
        nargs="+",
        type=int,
        default=DEFAULT_CENTERS,
        help="List of centers to evaluate."
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=DEFAULT_FOLDS,
        help="List of folds to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="Directory where evaluation CSVs will be written."
    )
    parser.add_argument(
        "--out_center",
        type=int,
        default=6,
        help="Held-out center for federated config."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Resize target size for evaluation images."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation dataloaders."
    )
    parser.add_argument(
        "--load_in_ram",
        action="store_true",
        help="Load images into RAM for evaluation."
    )
    parser.add_argument(
        "--load_names",
        action="store_true",
        help="Return image paths in dataset items for evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="polypgen",
        help="Dataset name used for inference thresholding."
    )
    return parser.parse_args()


def discover_checkpoint_sets(ckpt_dir):
    pattern = re.compile(
        r'^(?P<base>.+)_fold(?P<fold>\d+)(?:_(?P<variant>.+))?_(?P<status>best|best_small|last)\.pt$'
    )
    ckpt_sets = {}
    for fname in sorted(os.listdir(ckpt_dir)):
        match = pattern.match(fname)
        if not match:
            continue
        status = match.group('status')
        if status != 'best':
            continue

        base = match.group('base')
        fold = int(match.group('fold'))
        variant = match.group('variant')
        ckpt_name = f"{base}_{variant}" if variant else base
        ckpt_sets.setdefault(ckpt_name, {})[fold] = os.path.join(ckpt_dir, fname)

    if not ckpt_sets:
        raise RuntimeError(f"No best checkpoint files found in {ckpt_dir}.")

    return ckpt_sets


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


def save_matrix_with_means(df, metric_name, output_dir):
    matrix = df.copy()
    matrix["Mean"] = matrix.mean(axis=1)
    mean_row = matrix.mean(axis=0)
    mean_row.name = "Mean"
    matrix = pd.concat([matrix, pd.DataFrame([mean_row])])
    matrix.to_csv(os.path.join(output_dir, f"{metric_name}_matrix.csv"))
    return matrix


def print_matrix(metric_name, matrix):
    print(f"\n=== {metric_name.upper()} matrix ===")
    print(matrix.to_string(float_format="%.4f"))


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_sets = discover_checkpoint_sets(args.ckpt_dir)
    print(f"Discovered checkpoint sets: {list(checkpoint_sets.keys())}")

    wandb.init(
        project="evaluate_folds_new",
        config={
            "ckpt_dir": args.ckpt_dir,
            "centers": args.centers,
            "folds": args.folds,
            "out_center": args.out_center,
            "target_size": args.target_size,
            "batch_size": args.batch_size,
            "dataset": args.dataset,
        },
    )

    metric_names = ["dice", "iou"]
    centers = [f"Center{center}" for center in args.centers]

    results = {
        "dice": pd.DataFrame(index=[], columns=centers, dtype=float),
        "iou": pd.DataFrame(index=[], columns=centers, dtype=float),
    }

    flat_rows = []
    os.makedirs(args.output_dir, exist_ok=True)

    requested_folds = set(args.folds)
    for ckpt_name, ckpt_folds in checkpoint_sets.items():
        for fold_i in sorted(ckpt_folds.keys()):
            if fold_i not in requested_folds:
                continue

            ckpt_label = f"{ckpt_name}_fold{fold_i}"
            print(f"Evaluating checkpoint {ckpt_label}")

            for name in model_names:
                if name in ckpt_name:
                    model_name = name
                    break
            else:
                raise RuntimeError(f"Unable to infer model type from checkpoint name '{ckpt_name}'")

            net = get_model(model_name, in_channels=3, use_lightning=False, local=False)
            ckpt_path = ckpt_folds[fold_i]
            ckpt_state = load_checkpoint_state_dict(ckpt_path, device)
            net.load_state_dict(ckpt_state)
            net = net.to(device)
            net.eval()

            for center_nr in args.centers:
                dataset_config = {
                    "batch_size": args.batch_size,
                    "target_size": args.target_size,
                    "out_center": args.out_center,
                    "splits": "per_fold",
                    "seed": 1,
                    "load_in_ram": args.load_in_ram,
                    "load_names": args.load_names,
                    "fold_nr": fold_i,
                }

                dataloaders, num_examples = load_polypgen_centralized(center_nr, dataset_config)
                arr, scores, _ = test(net, dataloaders["in_test"], device, args.dataset)

                # Compute per-evaluation mean metrics from per-image scores
                # scores columns: [img_path, dsc, iou, recall, precision, f2, jaccard, ap, small_polyp, inv_area]
                dsc_vals       = [s[1] for s in scores if not np.isnan(s[1])]
                iou_vals       = [s[2] for s in scores if not np.isnan(s[2])]
                recall_vals    = [s[3] for s in scores if not np.isnan(s[3])]
                precision_vals = [s[4] for s in scores if not np.isnan(s[4])]
                f2_vals        = [s[5] for s in scores if not np.isnan(s[5])]
                jaccard_vals   = [s[6] for s in scores if not np.isnan(s[6])]
                ap_vals        = [s[7] for s in scores if not np.isnan(s[7])]

                dice_value      = float(np.mean(dsc_vals))      if dsc_vals       else 0.0
                iou_value       = float(np.mean(iou_vals))      if iou_vals       else 0.0
                recall_value    = float(np.mean(recall_vals))   if recall_vals    else 0.0
                precision_value = float(np.mean(precision_vals))if precision_vals else 0.0
                f2_value        = float(np.mean(f2_vals))       if f2_vals        else 0.0
                jaccard_value   = float(np.mean(jaccard_vals))  if jaccard_vals   else 0.0
                ap_value        = float(np.mean(ap_vals))       if ap_vals        else 0.0

                center_label = f"Center{center_nr}"

                flat_rows.append({
                    "framework":    ckpt_name,
                    "checkpoint":   ckpt_label,
                    "center":       center_label,
                    "fold":         fold_i,
                    "dice":         dice_value,
                    "iou":          iou_value,
                    "recall":       recall_value,
                    "precision":    precision_value,
                    "f2":           f2_value,
                    "jaccard":      jaccard_value,
                    "ap":           ap_value,
                    "n_images":     num_examples.get("in_test", 0),
                    "train_images": num_examples.get("train", 0),
                    "test_images":  num_examples.get("in_test", 0),
                })

    # Build flat dataframe with all per-evaluation results
    flat_df = pd.DataFrame(flat_rows)
    flat_csv_path = os.path.join(args.output_dir, "evaluation_flat_table.csv")
    flat_df.to_csv(flat_csv_path, index=False)

    # Save one CSV per framework
    all_summaries = []  # collect all framework summaries for a combined CSV
    frameworks = flat_df["framework"].unique()

    for framework in frameworks:
        fw_df = flat_df[flat_df["framework"] == framework]
        centers = sorted(fw_df["center"].unique(), key=lambda x: int(x.replace("Center", "")))

        rows = []
        for center in centers:
            subset = fw_df[fw_df["center"] == center]
            # Average across folds
            dice      = subset["dice"].mean()
            iou       = subset["iou"].mean()
            recall    = subset["recall"].mean()
            precision = subset["precision"].mean()
            f2        = subset["f2"].mean()
            jaccard   = subset["jaccard"].mean()
            ap        = subset["ap"].mean()
            # Dataset stats (same across folds for this center)
            n_images     = int(subset["n_images"].iloc[0])     if not subset.empty else 0
            train_images = int(subset["train_images"].iloc[0]) if not subset.empty else 0
            test_images  = int(subset["test_images"].iloc[0])  if not subset.empty else 0

            rows.append({
                "Framework":    framework,
                "Center":       center,
                "IoU":          iou,
                "Dice":         dice,
                "Recall":       recall,
                "Precision":    precision,
                "F2":           f2,
                "Jaccard":      jaccard,
                "AP":           ap,
                "Total_images": n_images,
                "Train_images": train_images,
                "Test_images":  test_images,
            })

        summary_df = pd.DataFrame(rows).set_index("Center")

        # Add Average row (mean across centers, numeric columns only)
        numeric_cols = summary_df.select_dtypes(include="number").columns
        avg_row = summary_df[numeric_cols].mean(axis=0)
        avg_row["Framework"] = framework
        avg_row.name = "Average"
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])])

        # Save per-framework CSV
        fw_csv_path = os.path.join(args.output_dir, f"{framework}_summary.csv")
        summary_df.to_csv(fw_csv_path)
        print(f"Saved {framework} summary to {os.path.abspath(fw_csv_path)}")

        # Collect for combined CSV
        all_summaries.append(summary_df.reset_index())

        # Print a publication-friendly table to terminal
        print(f"\nFramework: {framework}")
        if not flat_df.empty:
            first_center = centers[0] if centers else None
            if first_center is not None:
                info_subset = fw_df[fw_df["center"] == first_center]
                if not info_subset.empty:
                    print(f"PolypGen {first_center}")
                    print(f"Total images: {int(info_subset['n_images'].iloc[0])}")
                    print(f"Training images: {int(info_subset['train_images'].iloc[0])}")
                    print(f"Testing images: {int(info_subset['test_images'].iloc[0])}")

        print(
            summary_df[["IoU", "Dice", "Recall", "Precision", "F2", "Jaccard", "AP"]]
            .to_string(float_format="%.4f")
        )

    # Save combined summary of all frameworks into a single CSV
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_csv_path = os.path.join(args.output_dir, "all_frameworks_summary.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nSaved combined summary to {os.path.abspath(combined_csv_path)}")

    print(f"Saved flat table to {os.path.abspath(flat_csv_path)}")


def save_individual_scores(ckpt_type_name, dict_ckpt, fold_numbers):
    for scores_set in ["val_scores", "in_scores", "out_scores"]:
        df_rows = []
        for fold_i in fold_numbers:
            fold_scores = dict_ckpt[fold_i][scores_set]
            for score in fold_scores:
                df_rows.append([
                    fold_i,
                    score[0],   # img_path
                    score[1],   # dsc
                    score[2],   # iou
                    score[3],   # recall
                    score[4],   # precision
                    score[5],   # f2
                    score[6],   # jaccard
                    score[7],   # ap
                    score[8],   # small_polyp
                    score[9],   # inv_area
                ])
        df = pd.DataFrame(df_rows, columns=[
            "fold_nr", "img_path", "dsc", "iou", "recall", "precision",
            "f2", "jaccard", "ap", "small_polyp", "inv_area"
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

            dice_list      = [s[1] for s in scores if not np.isnan(s[1])]
            iou_list       = [s[2] for s in scores if not np.isnan(s[2])]
            recall_list    = [s[3] for s in scores if not np.isnan(s[3])]
            precision_list = [s[4] for s in scores if not np.isnan(s[4])]
            f2_list        = [s[5] for s in scores if not np.isnan(s[5])]
            jacc_list      = [s[6] for s in scores if not np.isnan(s[6])]
            ap_list        = [s[7] for s in scores if not np.isnan(s[7])]

            summary_rows.append([
                fold_i,
                split,
                len(scores),
                np.mean(dice_list)      if dice_list      else 0.0,
                np.mean(iou_list)       if iou_list       else 0.0,
                np.mean(recall_list)    if recall_list    else 0.0,
                np.mean(precision_list) if precision_list else 0.0,
                np.mean(f2_list)        if f2_list        else 0.0,
                np.mean(jacc_list)      if jacc_list      else 0.0,
                np.mean(ap_list)        if ap_list        else 0.0,
            ])

        df = pd.DataFrame(summary_rows, columns=[
            "fold_nr", "split", "n_images",
            "mean_dsc", "mean_iou", "mean_recall", "mean_precision",
            "mean_f2", "mean_jaccard", "mean_ap"
        ])
        df.to_csv(
            os.path.join("fold_results_fedAVG", f"{ckpt_type_name}_fold{fold_i}_summary.csv"),
            index=False
        )


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
            pred   = (pred == 1).float()
            target = (y_true == 1).float()

            prob = torch.softmax(out[0], dim=1)[:, 1, ...]

            batch_size_actual = images.size(0)
            for b in range(batch_size_actual):
                single_img_path = img_path[b]
                small_polyp, inv_area = check_small_polyp(single_img_path, True, dataset)

                single_pred   = pred[b]
                single_target = target[b]
                single_prob   = prob[b]

                intersection = (single_pred * single_target).sum()
                union        = ((single_pred + single_target) > 0).float().sum()

                dsc = (
                    (2.0 * intersection + 1e-7)
                    / (single_pred.sum() + single_target.sum() + 1e-7)
                ).item()
                iou = ((intersection + 1e-7) / (union + 1e-7)).item()

                true_positives  = (single_pred * single_target).sum().item()
                false_positives = (single_pred * (1 - single_target)).sum().item()
                false_negatives = ((1 - single_pred) * single_target).sum().item()

                precision = true_positives / (true_positives + false_positives + 1e-7)
                recall    = true_positives / (true_positives + false_negatives + 1e-7)
                f2        = (5 * precision * recall) / (4 * precision + recall + 1e-7)
                jaccard   = iou  # Jaccard == IoU for binary segmentation

                # Compute Average Precision for this single image
                if single_target.sum() > 0:
                    ap = average_precision_score(
                        single_target.detach().cpu().numpy().ravel(),
                        single_prob.detach().cpu().numpy().ravel()
                    )
                else:
                    ap = float('nan')

                dice_scores["dice"].append(dsc)

                if inv_area > 0:
                    dice_scores["dice_n"].append(dsc)

                    if small_polyp:
                        dice_scores["dice_s"].append(dsc)
                        if inv_area >= threshold:
                            dice_scores["dice_t"].append(dsc)
                    else:
                        dice_scores["dice_l"].append(dsc)

                # scores row: [img_path, dsc, iou, recall, precision, f2, jaccard, ap, small_polyp, inv_area]
                img_scores.append([
                    single_img_path,
                    dsc,
                    iou,
                    recall,
                    precision,
                    f2,
                    jaccard,
                    ap,
                    small_polyp,
                    inv_area,
                ])

    dice_s = np.mean(dice_scores['dice_s']) if dice_scores['dice_s'] else 0.0
    dice_t = np.mean(dice_scores['dice_t']) if dice_scores['dice_t'] else 0.0
    dice_l = np.mean(dice_scores['dice_l']) if dice_scores['dice_l'] else 0.0
    dice_n = np.mean(dice_scores['dice_n']) if dice_scores['dice_n'] else 0.0
    dice   = np.mean(dice_scores['dice'])   if dice_scores['dice']   else 0.0

    arr = np.array([dice_s, dice_t, dice_l, dice_n, dice])

    def get_len(lst):
        return len(lst)

    len_s   = get_len(dice_scores['dice_s'])
    len_t   = get_len(dice_scores['dice_t'])
    len_l   = get_len(dice_scores['dice_l'])
    len_n   = get_len(dice_scores['dice_n'])
    len_all = get_len(dice_scores['dice'])

    ratio_s_n   = (len_s / len_n)   if len_n   > 0 else 0.0
    ratio_t_s   = (len_t / len_s)   if len_s   > 0 else 0.0
    ratio_l_n   = (len_l / len_n)   if len_n   > 0 else 0.0
    ratio_n_all = (len_n / len_all) if len_all > 0 else 0.0

    ratios = (f"dice_s: {ratio_s_n:.4f} | "
              f"dice_t: {ratio_t_s:.4f} | "
              f"dice_l: {ratio_l_n:.4f} | "
              f"dice_n: {ratio_n_all:.4f}")

    return arr, img_scores, ratios


if __name__ == "__main__":
    main()