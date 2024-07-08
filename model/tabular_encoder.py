import torch
import torch.nn as nn

# class TabularEncoder(nn.Module):
#     def __init__(self, dict):
#         self.input_dim = dict['input_dim']
#         self.output_dim = dict['output_dim']
        
#         super(TabularEncoder, self).__init__()
        
#         self.model = TabNetEncoder(self.input_dim, self.output_dim)
        
#     def forward(self, x):
#         return self.model(x)
    
class TabularEncoder(nn.Module):
    def __init__(self, dict):
        self.input_dim = dict['input_dim']
        self.output_dim = dict['output_dim']
        
        super(TabularEncoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)