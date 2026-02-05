import os
import torch
import numpy as np
# TODO bisschen ordentlicher aber an sich optional

class early_stopping:
    """
    Early stooping to stop when val loss doesn't improve
    """

    def __init__(self, patience : int = 7, min_improve : float = 1e-3):
        """
        Parameters
        ----------
        patience : int
            Max number of epochs without an improvement
        
        min_improve :
            Min change in val loss, to qualify as improvement
        """
        
        self.patience    = patience
        self.min_improve = min_improve
        self.counter     = 0
        self.best_score  = 0
        self.early_stop  = False
        self.min_loss    = 
