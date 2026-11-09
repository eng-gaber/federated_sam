import os
import csv
import json
import time
import datetime
from typing import Dict, Any, List, Optional


class ExperimentTracker:
    """
    Tracks federated rounds, records in-domain and out-of-distribution metrics,
    appends to metrics_history.csv after each round, and dumps run_summary.json.
    """

    CSV_FIELDS = [
        "round",
        "timestamp",
        "in_domain_dsc",
        "in_domain_iou",
        "in_domain_pa",
        "in_domain_recall",
        "in_domain_hd95",
        "in_domain_dsc_small",
        "in_domain_dsc_large",
        "c6_dsc",
        "c6_iou",
        "c6_pa",
        "c6_recall",
        "c6_hd95",
        "c6_dsc_small",
        "c6_dsc_large",
        "best_c6_dsc_so_far",
        "best_c6_small_so_far",
    ]

    def __init__(self, run_dir: str, run_name: str, config: Dict[str, Any]):
        self.run_dir = run_dir
        self.run_name = run_name
        self.config = config
        self.start_time = time.time()
        self.csv_path = os.path.join(run_dir, "metrics_history.csv")
        self.summary_path = os.path.join(run_dir, "run_summary.json")

        self.history: List[Dict[str, Any]] = []
        self.best_c6_dsc = 0.0
        self.best_c6_dsc_round = -1
        self.best_c6_small_dsc = 0.0
        self.best_c6_small_round = -1

        # Load existing history if resuming, or initialize CSV header
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.history.append(row)
                    c6_dsc = float(row.get("c6_dsc", 0.0))
                    c6_small = float(row.get("c6_dsc_small", 0.0))
                    r_idx = int(row.get("round", -1))
                    if c6_dsc > self.best_c6_dsc:
                        self.best_c6_dsc = c6_dsc
                        self.best_c6_dsc_round = r_idx
                    if c6_small > self.best_c6_small_dsc:
                        self.best_c6_small_dsc = c6_small
                        self.best_c6_small_round = r_idx
        else:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def log_round(
        self,
        round_idx: int,
        in_domain_metrics: Dict[str, float],
        c6_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Record metrics for a completed communication round and update CSV."""
        c6_dsc = float(c6_metrics.get("dice", 0.0))
        c6_small = float(c6_metrics.get("dice_small", 0.0))

        if c6_dsc > self.best_c6_dsc:
            self.best_c6_dsc = c6_dsc
            self.best_c6_dsc_round = round_idx
            self.best_c6_metrics = {
                "round": round_idx,
                "dice": round(c6_dsc, 4),
                "dice_small": round(c6_small, 4),
                "dice_large": round(float(c6_metrics.get("dice_large", 0.0)), 4),
                "iou": round(float(c6_metrics.get("iou", 0.0)), 4),
                "recall": round(float(c6_metrics.get("recall", 0.0)), 4),
                "pa": round(float(c6_metrics.get("pa", 0.0)), 4),
                "hd95": round(float(c6_metrics.get("hd95", 0.0)), 2),
            }

        if c6_small > self.best_c6_small_dsc:
            self.best_c6_small_dsc = c6_small
            self.best_c6_small_round = round_idx

        record = {
            "round": round_idx,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "in_domain_dsc": round(float(in_domain_metrics.get("dice", 0.0)), 4),
            "in_domain_iou": round(float(in_domain_metrics.get("iou", 0.0)), 4),
            "in_domain_pa": round(float(in_domain_metrics.get("pa", 0.0)), 4),
            "in_domain_recall": round(float(in_domain_metrics.get("recall", 0.0)), 4),
            "in_domain_hd95": round(float(in_domain_metrics.get("hd95", 0.0)), 2),
            "in_domain_dsc_small": round(float(in_domain_metrics.get("dice_small", 0.0)), 4),
            "in_domain_dsc_large": round(float(in_domain_metrics.get("dice_large", 0.0)), 4),
            "c6_dsc": round(c6_dsc, 4),
            "c6_iou": round(float(c6_metrics.get("iou", 0.0)), 4),
            "c6_pa": round(float(c6_metrics.get("pa", 0.0)), 4),
            "c6_recall": round(float(c6_metrics.get("recall", 0.0)), 4),
            "c6_hd95": round(float(c6_metrics.get("hd95", 0.0)), 2),
            "c6_dsc_small": round(c6_small, 4),
            "c6_dsc_large": round(float(c6_metrics.get("dice_large", 0.0)), 4),
            "best_c6_dsc_so_far": round(self.best_c6_dsc, 4),
            "best_c6_small_so_far": round(self.best_c6_small_dsc, 4),
        }

        self.history.append(record)

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(record)

        # Update summary JSON in real-time
        self.save_summary(status="running")
        return record

    def save_summary(self, status: str = "completed") -> str:
        """Write a comprehensive summary of the experiment run to JSON."""
        elapsed_sec = time.time() - self.start_time
        summary = {
            "run_name": self.run_name,
            "status": status,
            "total_rounds_completed": len(self.history),
            "elapsed_seconds": round(elapsed_sec, 2),
            "elapsed_formatted": str(datetime.timedelta(seconds=int(elapsed_sec))),
            "best_overall_unseen_c6": getattr(self, "best_c6_metrics", {
                "round": self.best_c6_dsc_round,
                "dice": round(self.best_c6_dsc, 4),
                "dice_small": round(self.best_c6_small_dsc, 4),
            }),
            "best_small_unseen_c6": {
                "round": self.best_c6_small_round,
                "dice_small": round(self.best_c6_small_dsc, 4),
            },
            "last_round_metrics": self.history[-1] if self.history else {},
            "checkpoints": {
                "best": os.path.join(self.run_dir, "FedFGS_SAM_best.pth"),
                "best_small": os.path.join(self.run_dir, "FedFGS_SAM_best_small.pth"),
                "last": os.path.join(self.run_dir, "FedFGS_SAM_last.pth"),
            },
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return self.summary_path
