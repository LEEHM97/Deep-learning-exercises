import argparse
from xml.etree.ElementInclude import default_loader

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('__model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_abailale() else -1)
    
    p.add_argument('--train_ratio', type=float, default=.8)
    
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    
    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)
    
    p.add_argument('--verbose', type=int, default=1)
    
    config = p.parse_args()
    
    return config