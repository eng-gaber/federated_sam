"""
Federated Loss Functions for Fed-FGS.
Re-exports and encapsulates the composite Dice + BCE + Focal loss.
"""

from fedfgs_loss import FedFGSCompositeLoss

__all__ = ["FedFGSCompositeLoss"]
