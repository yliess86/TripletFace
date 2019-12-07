"""model.py

The file contains the model definition for triplet loss facial recognition based
on pretrained resnet18.
"""
import torch.nn as nn
import torch

from torchvision import models

class Encoder( nn.Module ):
    """Encoder

    The class describe the model architecture.

    Attributes
    ----------
    z_size: int
            size of the latent space ( output size )
    resnet: nn.Module
            finetuned resnet model ( pretrained with fc replaced )
    """

    def __init__( self: 'Encoder', z_size: int ) -> None:
        """__init__

        Initialize the model architecture.

        Parameters
        ----------
        z_size: int
                size of the latent space ( output size )
        """
        super( Encoder, self ).__init__( )
        self.z_size    = z_size
        self.resnet    = models.resnet18( pretrained = True )
        self._disable_grad( )
        n_features     = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear( n_features, z_size )

    def _disable_grad( self: 'Encoder' ) -> None:
        """_disable_grad

        Disable all gradients for the model to freeze resnet for finetuning.
        Operation needs to happen before insertion of the new fc module.
        """
        for param in self.resnet.parameters( ):
            param.requires_grad = False

    def forward( self: 'Encoder', X: torch.Tensor ) -> torch.Tensor:
        """forward

        Forward path of the model. Outputs the z_size latent vector for each
        image of the X tensor.

        Parameters
        ----------
        X: torch.Tensor
           Input tensor [ Batch, Chanel, Height, Width ]

        Returns
        -------
        X: torch.Tensor
           Output tensor [ Batch, z_size ]
        """
        return self.resnet( X )
