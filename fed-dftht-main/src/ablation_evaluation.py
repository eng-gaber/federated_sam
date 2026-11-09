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
CHECKPOINT_FILENAME = "federated_sd_utnet_p_factor_fold5_dftht_best.pt"
MODEL_NAME = "sd_utnet"

DEFAULT_FOLD = 5
DEFAULT_OUT_CENTER = 6



def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fold 5 federated SD-UTNet checkpoint on out-center 6 PolypGen data."
    )
    parser.add_argument(
        "--ckpt_dir",
        default=DEFAULT_CKPT_DIR,
        help="Checkpoint directory containing the fold 5 model."
    )
    parser.add_argument(
        "--ckpt_filename",
        default=CHECKPOINT_FILENAME,
        help="Checkpoint filename to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_results_out_center6",
        help="Directory where evaluation CSVs will be written."
    )
    parser.add_argument(
        "--out_center",
        type=int,
        default=DEFAULT_OUT_CENTER,
        help="Held-out center to evaluate on."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=DEFAULT_FOLD,
        help="Fold number (used for data split config)."
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
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    #  Build full checkpoint path and validate it exists
    # ------------------------------------------------------------------ #
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_filename)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {os.path.abspath(ckpt_path)}"
        )
    print(f"Using checkpoint : {os.path.abspath(ckpt_path)}")
    print(f"Model            : {MODEL_NAME}")
    print(f"Fold             : {args.fold}")
    print(f"Out-center       : {args.out_center}")

    # ------------------------------------------------------------------ #
    #  Initialise W&B
    # ------------------------------------------------------------------ #
    wandb.init(
        project="evaluate_outcenter6",
        config={
            "ckpt_path":   ckpt_path,
            "model":       MODEL_NAME,
            "fold":        args.fold,
            "out_center":  args.out_center,
            "target_size": args.target_size,
            "batch_size":  args.batch_size,
            "dataset":     args.dataset,
        },
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Load model
    # ------------------------------------------------------------------ #
    net = get_model(MODEL_NAME, in_channels=3, use_lightning=False, local=False)
    ckpt_state = load_checkpoint_state_dict(ckpt_path, device)
    net.load_state_dict(ckpt_state)
    net = net.to(device)
    net.eval()
    print("Model loaded successfully.")

        # ------------------------------------------------------------------ #
    #  Load out-center 6 dataloader via federated config
    # ------------------------------------------------------------------ #
    dataset_config = {
        "batch_size":   args.batch_size,
        "target_size":  args.target_size,
        "out_center":   args.out_center,
        "splits":       "per_fold",
        "seed":         1,
        "load_in_ram":  args.load_in_ram,
        "load_names":   args.load_names,
        "fold_nr":      args.fold,
    }

    # Pass center_nr=None so the function builds the out_test loader
    # from data_config["out_center"] (= args.out_center, e.g. 6).
    dataloaders, num_examples = load_polypgen_federated(
        center_nr=None,
        data_config=dataset_config,
    )

    out_test_key = "out_test"
    if out_test_key not in dataloaders or dataloaders[out_test_key] is None:
        raise KeyError(
            f"'out_test' loader not available. "
            f"Available keys: {list(dataloaders.keys())}"
        )

    n_out_test = num_examples.get(out_test_key, 0)
    if n_out_test == 0:
        raise RuntimeError(
            f"Out-center {args.out_center} test set is empty. "
            f"Check that data_config['out_center'] is set correctly."
        )

    print(f"Using dataloader key '{out_test_key}' for out-center {args.out_center}.")
    print(f"Number of out-center test images: {n_out_test}")

    # ------------------------------------------------------------------ #
    #  Run evaluation
    # ------------------------------------------------------------------ #
    arr, img_scores, ratios = test(
        net, dataloaders[out_test_key], device, args.dataset
    )

    print(f"\nPolyp size ratios: {ratios}")
    print(f"Aggregate dice array [dice_s, dice_t, dice_l, dice_n, dice]: {arr}")

    # ------------------------------------------------------------------ #
    #  Collect per-image metrics
    # ------------------------------------------------------------------ #
    dsc_vals       = [s[1] for s in img_scores if not np.isnan(s[1])]
    iou_vals       = [s[2] for s in img_scores if not np.isnan(s[2])]
    recall_vals    = [s[3] for s in img_scores if not np.isnan(s[3])]
    precision_vals = [s[4] for s in img_scores if not np.isnan(s[4])]
    f2_vals        = [s[5] for s in img_scores if not np.isnan(s[5])]
    jaccard_vals   = [s[6] for s in img_scores if not np.isnan(s[6])]
    ap_vals        = [s[7] for s in img_scores if not np.isnan(s[7])]

    dice_value      = float(np.mean(dsc_vals))       if dsc_vals       else 0.0
    iou_value       = float(np.mean(iou_vals))       if iou_vals       else 0.0
    recall_value    = float(np.mean(recall_vals))    if recall_vals    else 0.0
    precision_value = float(np.mean(precision_vals)) if precision_vals else 0.0
    f2_value        = float(np.mean(f2_vals))        if f2_vals        else 0.0
    jaccard_value   = float(np.mean(jaccard_vals))   if jaccard_vals   else 0.0
    ap_value        = float(np.mean(ap_vals))        if ap_vals        else 0.0

    # ------------------------------------------------------------------ #
    #  Print summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print(f"  Evaluation Summary — Center {args.out_center} (out-center)")
    print("=" * 55)
    print(f"  Checkpoint : {args.ckpt_filename}")
    print(f"  Fold       : {args.fold}")
    print(f"  Images     : {len(img_scores)}")
    print("-" * 55)
    print(f"  Dice       : {dice_value:.4f}")
    print(f"  IoU        : {iou_value:.4f}")
    print(f"  Recall     : {recall_value:.4f}")
    print(f"  Precision  : {precision_value:.4f}")
    print(f"  F2         : {f2_value:.4f}")
    print(f"  Jaccard    : {jaccard_value:.4f}")
    print(f"  AP         : {ap_value:.4f}")
    print("=" * 55)

    # ------------------------------------------------------------------ #
    #  Log to W&B
    # ------------------------------------------------------------------ #
    wandb.log({
        "out_center":  args.out_center,
        "fold":        args.fold,
        "n_images":    len(img_scores),
        "dice":        dice_value,
        "iou":         iou_value,
        "recall":      recall_value,
        "precision":   precision_value,
        "f2":          f2_value,
        "jaccard":     jaccard_value,
        "ap":          ap_value,
        # aggregate dice sub-groups
        "dice_s":      float(arr[0]),
        "dice_t":      float(arr[1]),
        "dice_l":      float(arr[2]),
        "dice_n":      float(arr[3]),
        "dice_all":    float(arr[4]),
    })

    # ------------------------------------------------------------------ #
    #  Save per-image scores CSV
    # ------------------------------------------------------------------ #
    per_image_df = pd.DataFrame(img_scores, columns=[
        "img_path", "dsc", "iou", "recall", "precision",
        "f2", "jaccard", "ap", "small_polyp", "inv_area"
    ])
    per_image_csv = os.path.join(
        args.output_dir,
        f"per_image_scores_center{args.out_center}_fold{args.fold}.csv"
    )
    per_image_df.to_csv(per_image_csv, index=False)
    print(f"\nPer-image scores saved to : {os.path.abspath(per_image_csv)}")

    # ------------------------------------------------------------------ #
    #  Save summary CSV
    # ------------------------------------------------------------------ #
    summary_df = pd.DataFrame([{
        "checkpoint":   args.ckpt_filename,
        "model":        MODEL_NAME,
        "fold":         args.fold,
        "center":       f"Center{args.out_center}",
        "split":        "out_test",
        "n_images":     len(img_scores),
        "dice":         dice_value,
        "iou":          iou_value,
        "recall":       recall_value,
        "precision":    precision_value,
        "f2":           f2_value,
        "jaccard":      jaccard_value,
        "ap":           ap_value,
        "dice_s":       float(arr[0]),
        "dice_t":       float(arr[1]),
        "dice_l":       float(arr[2]),
        "dice_n":       float(arr[3]),
    }])
    summary_csv = os.path.join(
        args.output_dir,
        f"summary_center{args.out_center}_fold{args.fold}.csv"
    )
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to           : {os.path.abspath(summary_csv)}")

    wandb.finish()


# ------------------------------------------------------------------ #
#  test() — unchanged from original
# ------------------------------------------------------------------ #
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