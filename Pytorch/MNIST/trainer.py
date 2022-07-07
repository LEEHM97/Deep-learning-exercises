from copy import deepcopy
import enum
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
            
        x = x.split(batch_size, dim=0)    
        y = y.split(batch_size, dim=0)    
        
        return x, y
        
    def _train(self, x, y, config):
        self.model.train()
        
        x, y = self.batchfy(x, y, config.batch_size)
        total_loss = 0
        
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            
            # Initialize the gradients of the model
            self.optimizer.zero_grad()
            loss_i.backward()
            
            self.optimizer.step()
            
            if config.Verbose >= 2:
                print("Train Iteration (%d/%d): loss=%.4e" %(i+1, len(x), float(loss_i)))
            
            # Detach to prevent memory leak 
            total_loss += float(loss_i)
        return total_loss / len(x)
    
    def _validate(self, x, y, config):
        # 
    
    
    
    
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