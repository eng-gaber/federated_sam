import os
import json
import yaml
import datetime
from typing import Dict, Any, Tuple


def flatten_nested_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Flatten nested config dictionaries into a flat key-value dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data or {}


def setup_experiment_run(args: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve run directory, generate run_name if needed, and serialize effective config to config.json.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = getattr(args, "run_name", None)
    if not run_name:
        rank = getattr(args, "lora_rank", 4)
        rounds = getattr(args, "num_rounds", 10)
        run_name = f"run_{timestamp}_r{rank}_{rounds}rounds"

    output_root = getattr(args, "output_root", "checkpoints/runs")
    run_dir = getattr(args, "output_dir", None)
    if not run_dir:
        run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    args.run_name = run_name
    args.run_dir = run_dir
    args.output_dir = run_dir

    # Save effective configuration as config.json
    all_args_dict = {
        k: str(v) if isinstance(v, (datetime.datetime, os.PathLike)) else v
        for k, v in vars(args).items()
        if not k.startswith("_") and not callable(v)
    }
    clean_config = {}
    for k, v in all_args_dict.items():
        try:
            json.dumps(v)
            clean_config[k] = v
        except (TypeError, OverflowError):
            clean_config[k] = str(v)

    clean_config["_created_at"] = datetime.datetime.now().isoformat()
    clean_config["run_dir"] = run_dir
    clean_config["run_name"] = run_name

    save_config_json(run_dir, clean_config)
    print(f"📁 Initialized experiment run directory: {run_dir}")
    print(f"💾 Saved run configuration to: {os.path.join(run_dir, 'config.json')}")

    return run_dir, clean_config


def save_config_json(directory: str, config: Dict[str, Any]) -> str:
    """Save configuration dictionary to a JSON file."""
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, "config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return out_path
