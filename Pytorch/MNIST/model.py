import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularaizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularaizer(use_batch_norm, output_size),
        )
    
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        
        # |y| = (batch_size, output_size)
        return y