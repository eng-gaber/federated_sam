import torch
from models import get_model
from data import load_polypgen_federated
import torchmetrics
import os
import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

CKPTS = [
    "federated_sd_utnet_fed_exp2_best.pt",
    "federated_sd_utnet_p_factor_fold5_last.pt"
]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_config = {
        "batch_size": 4,
        "target_size": 512,
        "out_center": 6,
        "splits": "per_fold", # we can use per_fold fold_nr=5 or per_patient
        "seed": 1,
        "load_in_ram": False,
        "load_names": False,
        "fold_nr": 5
    }

    # Load data
    dataloaders, num_examples = load_polypgen_federated(None, data_config=data_config)
    print("Num examples:", num_examples)

    for ckpt_name in CKPTS:
        print(f"\n============================\nEvaluating {ckpt_name}\n============================")
        
        path = os.path.join("ckpts", ckpt_name)
        if not os.path.exists(path):
            print(f"File {path} not found. Skipping...")
            continue
            
        model_names = ["sd_utnet_fed", "sd_utnet", "sdnet", "unet", "utnet"]
        model_name = "sd_utnet"
        for name in model_names:
            if name in ckpt_name:
                model_name = name
                break
                
        in_channels = 3
        net = get_model(model_name, in_channels=in_channels, use_lightning=False, local=False)
        
        ckpt = torch.load(path, map_location=device)
        # Handle state_dict formatting
        if 'state_dict' in ckpt:
            net.load_state_dict(ckpt['state_dict'])
        else:
            net.load_state_dict(ckpt)
            
        net = net.to(device)
        net.eval()
        
        # Test on in_test to get client-specific results
        dice_scores_centers = defaultdict(lambda: {'dsc': list(), 'iou': list()})
        
        with torch.no_grad():
            for images, y_true, img_path, _ in dataloaders['in_test']:
                images, y_true = images.to(device), y_true.to(device)

                # Extracts center_nr (usually 1 to 5)
                # the path format is data/data_C1/images_C1/file.jpg or similar
                center_str = img_path[0].split('data_C')[-1][0]
                center_nr = int(center_str)

                out = net(images)
                out = out if type(out) is tuple else tuple([out])
                
                # DSC
                dsc = torchmetrics.functional.classification.binary_f1_score(out[0][:, 1:, ...], y_true, zero_division=1.0)
                dsc = dsc.detach().cpu().item()
                
                # IOU
                iou = torchmetrics.functional.classification.binary_jaccard_index(out[0][:, 1:, ...], y_true)
                iou = iou.detach().cpu().item()

                dice_scores_centers[center_nr]["dsc"].append(dsc)
                dice_scores_centers[center_nr]["iou"].append(iou)

        for center_nr in sorted(dice_scores_centers.keys()):
            scores = dice_scores_centers[center_nr]
            mean_dsc = np.mean(scores['dsc'])
            mean_iou = np.mean(scores['iou'])
            num_imgs = len(scores['dsc'])
            print(f"Client {center_nr}: DSC = {mean_dsc:.4f}, IOU = {mean_iou:.4f} (from {num_imgs} images)")
        
        # Also test on out_test (Center 6)
        if 'out_test' in dataloaders and len(dataloaders['out_test']) > 0:
            out_scores = {'dsc': list(), 'iou': list()}
            with torch.no_grad():
                for images, y_true, img_path, _ in dataloaders['out_test']:
                    images, y_true = images.to(device), y_true.to(device)
                    out = net(images)
                    out = out if type(out) is tuple else tuple([out])
                    dsc = torchmetrics.functional.classification.binary_f1_score(out[0][:, 1:, ...], y_true, zero_division=1.0).item()
                    iou = torchmetrics.functional.classification.binary_jaccard_index(out[0][:, 1:, ...], y_true).item()
                    out_scores['dsc'].append(dsc)
                    out_scores['iou'].append(iou)
            print(f"Center 6 (Unseen): DSC = {np.mean(out_scores['dsc']):.4f}, IOU = {np.mean(out_scores['iou']):.4f} (from {len(out_scores['dsc'])} images)")

if __name__ == '__main__':
    main()
