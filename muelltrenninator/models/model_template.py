import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class model_template(nn.Module):

    def forward(self, x): 
        """
        An abstract method, that needs to be overwritten in the subclass
        """

        raise NotImplementedError

    def get_num_params(self) -> int: 
        """
        Returns
        -------
        the number of params of the supplied CNN : int
        """

        return sum(p.numel() for p in self.parameters())

    def get_name(self) -> str:
        """
        Returns 
        --------
        the class name : string
        """

        return self.__class__.__name__
