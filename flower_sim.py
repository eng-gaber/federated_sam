#!/usr/bin/env python
"""
Top-level entrypoint for Federated Simulation.
Delegates to the federated package (federated/flower_sim.py).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from federated.flower_sim import main

if __name__ == "__main__":
    main()
