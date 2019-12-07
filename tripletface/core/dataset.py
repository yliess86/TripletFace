"""dataset.py

The file contains all the code to load our dataset for triplet loss facial
recogintion.
"""
import numpy as np
import os

from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
from PIL import Image

class ImageFolder( Dataset ):
    """ImageFolder

    The class can load our dataset while providing labels and data speratly.

    Attributes
    ----------
    root  : str
            path to the folder containing the dataset
    labels: List[ int ]
            list of all the labels from the dataset
    files : List[ str ]
            list of all the paths to the image files from the dataset
    size  : int
            dataset size
    trans : transforms.transforms
            transformations to be applied before returning image data
    """

    def __init__( self: 'ImageFolder', root: str, trans: transforms.transforms ) -> None:
        """__init__

        Initialize dataset by importing labels and data pairs.

        Parameters
        ----------
        root : str
               path to the folder containing the dataset
        trans: transforms.transforms
               transformations to be applied before returning image data
        """
        self.root  = root
        self._get_pairs( )
        self.size  = len( self.files )
        self.trans = trans

    def _get_pairs( self: 'ImageFolder' ) -> None:
        """_get_pairs

        Import all labels and data pairs.
        """
        assert os.path.isdir( self.root ), f'The root { self.root } does not exists'
        names       = [ dir
            for dir in os.listdir( self.root )\
            if os.path.isdir( os.path.join( self.root, dir ) )
        ]

        self.labels = [ ]
        self.files  = [ ]
        for name in names:
            src   = os.path.join( self.root, name )
            files = [ os.path.join( src, file )
                for file in os.listdir( src )\
                if '.png' in file.lower( )
            ]
            for file in files:
                self.labels.append( int( name ) )
                self.files.append( file )

    def __len__( self: 'ImageFolder' ) -> int:
        """__len__

        Returns the dataset length ( number of samples )

        Returns
        -------
        size: int
              dataset size
        """
        return self.size

    def __getitem__( self: 'ImageFolder', idx: int ) -> Tuple[ int, np.ndarray ]:
        """__getitem__

        Returns label and data pairs at a given index.

        Parameters
        ----------
        idx: int
             index of the sample

        Returns
        -------
        label: int
               label of the sample at index idx
        X    : np.ndarray
               transformed image sample from file at index idx
        """
        label = self.labels[ idx ]
        img   = Image.open( self.files[ idx ] ).convert( 'RGB' )
        X     = self.trans( img )

        return label, X.numpy( )
