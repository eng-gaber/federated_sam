import os
import random

DATA_ROOT   = "/home/omanchat-1/Documents/fed/fed_sam/finetune-SAM/polyp_db_dataset/data"
CLIENTS     = ["C1", "C2", "C3", "C4", "C5", "C6"]
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
SEED        = 42

random.seed(SEED)

for client in CLIENTS:
    img_dir  = os.path.join(DATA_ROOT, f"data_{client}", f"images_{client}")
    mask_dir = os.path.join(DATA_ROOT, f"data_{client}", f"masks_{client}")
    out_dir  = os.path.join(DATA_ROOT, f"data_{client}")
    # base folder = what we pass as img_folder and mask_folder to the script
    base_dir = os.path.join(DATA_ROOT, f"data_{client}")

    if not os.path.exists(img_dir):
        print(f"SKIPPING {client} - not found: {img_dir}")
        continue

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\n{client}: {len(images)} images")

    pairs   = []
    missing = 0

    for img_name in images:
        base, ext = os.path.splitext(img_name)
        candidates = [
            img_name,
            f"{base}_mask{ext}",
            f"{base}_mask.png",
            f"mask_{img_name}",
        ]
        found = False
        for candidate in candidates:
            mask_full = os.path.join(mask_dir, candidate)
            if os.path.exists(mask_full):
                # relative paths from base_dir (data_C1/)
                img_rel  = os.path.join(f"images_{client}", img_name)
                mask_rel = os.path.join(f"masks_{client}",  candidate)
                pairs.append((img_rel, mask_rel))
                found = True
                break
        if not found:
            missing += 1

    print(f"  Matched: {len(pairs)} pairs, Missing: {missing}")

    random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val"  : pairs[n_train:n_train + n_val],
        "test" : pairs[n_train + n_val:]
    }

    for split_name, split_pairs in splits.items():
        csv_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            for ip, mp in split_pairs:
                f.write(f"{ip},{mp}\n")
        print(f"  {split_name}.csv -> {len(split_pairs)} pairs")

print("\nAll CSVs generated!")
