"""
Modular Federated Simulation Runner for Fed-FGS.
"""

import os
import sys
from typing import Dict, Any, List

import numpy as np
import torch

from .config import setup_experiment_run
from .tracker import ExperimentTracker


def execute_federated_simulation(args, run_sim_fn):
    """
    Standard wrapper around simulation function:
    - Sets up experiment run directory and config serialization
    - Initializes ExperimentTracker
    - Executes federated training
    - Finalizes run summary
    """
    run_dir, config_dict = setup_experiment_run(args)
    tracker = ExperimentTracker(run_dir=run_dir, run_name=args.run_name, config=config_dict)
    
    try:
        run_sim_fn(args, tracker=tracker)
        tracker.save_summary(status="completed")
    except Exception as e:
        tracker.save_summary(status=f"failed: {str(e)}")
        raise e

    return run_dir, tracker
