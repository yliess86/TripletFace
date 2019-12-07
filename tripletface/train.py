"""train.py

The file contains all the code to launch a training for the triplet loss facial
recognition model and can be call with python3 -m tripletface.train --help.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import os

from triplettorch import HardNegativeTripletMiner
from tripletface.core.dataset import ImageFolder
from tripletface.core.model import Encoder
from torch.utils.data import DataLoader
from triplettorch import TripletDataset
from torchvision import transforms
from sklearn.manifold import TSNE
from torch.optim import Adam
from tqdm import tqdm

"""argparse

This part describes all the options the module can be executed with.
"""
parser        = argparse.ArgumentParser( )
parser.add_argument( '-s', '--dataset_path',  type = str,   required = True )
parser.add_argument( '-m', '--model_path',    type = str,   required = True )
parser.add_argument( '-i', '--input_size',    type = int,   default  = 224 )
parser.add_argument( '-z', '--latent_size',   type = int,   default  = 64 )
parser.add_argument( '-b', '--batch_size',    type = int,   default  = 32 )
parser.add_argument( '-e', '--epochs',        type = int,   default  = 10 )
parser.add_argument( '-l', '--learning_rate', type = float, default  = 1e-3 )
parser.add_argument( '-w', '--n_workers',     type = int,   default  = 4 )
parser.add_argument( '-r', '--n_samples',     type = int,   default  = 6 )
args          = parser.parse_args( )

dataset_path  = args.dataset_path
model_path    = args.model_path

input_size    = args.input_size
latent_size   = args.latent_size

batch_size    = args.batch_size
epochs        = args.epochs
learning_rate = args.learning_rate
n_workers     = args.n_workers
n_samples     = args.n_samples
noise         = 1e-2

if not os.path.isdir( model_path ):
    os.mkdir( model_path )

"""trans

This part descibes all the transformations applied to the images for training
and testing.
"""
trans         = {
    'train': transforms.Compose( [
        transforms.RandomRotation( degrees = 360 ),
        transforms.Resize( size = input_size ),
        transforms.RandomCrop( size = input_size ),
        transforms.RandomVerticalFlip( p = 0.5 ),
        transforms.ColorJitter( brightness = .2, contrast = .2, saturation = .2, hue = .1 ),
        transforms.ToTensor( ),
        transforms.Lambda( lambda X: X * ( 1. - noise ) + torch.randn( X.shape ) * noise ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] ),
    'test':transforms.Compose( [
        transforms.Resize( size = input_size ),
        transforms.CenterCrop( size = input_size ),
        transforms.ToTensor( ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] )
}

"""folder

This part descibes all the folder dataset for training and testing.
"""
folder        = {
    'train': ImageFolder( os.path.join( dataset_path, 'train' ), trans[ 'train' ] ),
    'test' : ImageFolder( os.path.join( dataset_path, 'test'  ), trans[ 'test'  ] )
}

"""dataset

This part descibes all the triplet dataset for training and testing.
"""
dataset       = {
    'train': TripletDataset(
        np.array( folder[ 'train' ].labels ),
        lambda i: folder[ 'train' ][ i ][ 1 ],
        len( folder[ 'train' ] ),
        n_samples
    ),
    'test' : TripletDataset(
        np.array( folder[ 'test'  ].labels ),
        lambda i: folder[ 'test'  ][ i ][ 1 ],
        len( folder[ 'test'  ] ),
        1
    )
}

"""loader

This part descibes all the dataset loaders for training and testing.
"""
loader        = {
    'train': DataLoader( dataset[ 'train' ],
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = n_workers,
        pin_memory  = True
    ),
    'test' : DataLoader( dataset[ 'test' ],
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = n_workers,
        pin_memory  = False
    ),
}

"""encoder

This part contains the model, loss, optimizer and figure ( to plot ) used for
training.
"""
encoder       = Encoder( latent_size ).cuda( )
miner         = HardNegativeTripletMiner( .5 ).cuda( )
optim         = Adam( encoder.parameters( ), lr = learning_rate )

fig           = plt.figure( figsize = ( 8, 8 ) )
ax            = fig.add_subplot( 111 )

"""train

This part contains the training logic loop.
"""
for e in tqdm( range( epochs ), desc = 'Epoch' ):
    # ================== TRAIN ========================
    train_n        = len( loader[ 'train' ] )
    train_loss     = 0.
    train_frac_pos = 0.

    with tqdm( loader[ 'train' ], desc = 'Batch' ) as b_pbar:
        for b, batch in enumerate( b_pbar ):
            optim.zero_grad( )

            labels, data    = batch
            labels          = torch.cat( [ label for label in labels ], axis = 0 )
            data            = torch.cat( [ datum for datum in   data ], axis = 0 )
            labels          = labels.cuda( )
            data            = data.cuda( )

            embeddings      = encoder( data )
            loss, frac_pos  = miner( labels, embeddings )

            loss.backward( )
            optim.step( )

            train_loss     += loss.detach( ).item( )
            train_frac_pos += frac_pos.detach( ).item( ) if frac_pos is not None else \
                              0.

            b_pbar.set_postfix(
                train_loss     = train_loss / train_n,
                train_frac_pos = f'{( train_frac_pos / train_n ):.2%}'
            )

    # ================== TEST ========================
    if latent_size >= 2:
        test_embeddings = [ ]
        test_labels     = [ ]

        for b, batch in enumerate( tqdm( loader[ 'test' ], desc = 'Plot' ) ):
            labels, data = batch
            data         = torch.cat( [ datum for datum in   data ], axis = 0 )
            labels       = torch.cat( [ label for label in labels ], axis = 0 )
            embeddings   = encoder( data.cuda( ) ).detach( ).cpu( ).numpy( )
            labels       = labels.numpy( )

            test_embeddings.append( embeddings )
            test_labels.append( labels )

        test_embeddings = np.concatenate( test_embeddings, axis = 0 )
        test_labels     = np.concatenate(     test_labels, axis = 0 )

        if latent_size > 2:
            test_embeddings = TSNE( n_components = 2 ).fit_transform( test_embeddings )

        ax.clear( )
        ax.scatter(
            test_embeddings[ :, 0 ],
            test_embeddings[ :, 1 ],
            c = test_labels
        )

        fig.canvas.draw( )
        fig.savefig( os.path.join( model_path, f'vizualisation_{e}.png' ) )

    # ================== SAVE ========================
    torch.save( {
        'model': encoder.state_dict( ),
        'optim': optim.state_dict( ),
        'epoch': e,
    }, os.path.join( model_path, 'model.pt' ) )
