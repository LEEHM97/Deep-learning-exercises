from copy import deepcopy
from textwrap import indent
import numpy as np
import torch

class Trainer():
    def __init__self(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        
        super().__init__()
        
    def _batchfy(self, x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)
            
            return x, y
        
    def _train(self, x, y, config):
        pass
    
    def _validate(self, x, y, config):
        pass
    
    
    
    
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None
        
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate (valid_data[0], valid_data[1], config)
            
            # Use deep copy to take a snapshot of current best weighs
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                
            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" %(
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss
            ))
            
        # Restore to best model
        self.model.load_state_dict(best_model)