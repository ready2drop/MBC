import torch.nn as nn

    
class TabularEncoder_earlyfusion(nn.Module):
    def __init__(self, dict):
        self.input_dim = dict['input_dim']
        
        super(TabularEncoder_earlyfusion, self).__init__()
        
        self.model = nn.Sequential(
            # nn.Linear(self.input_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, self.input_dim),
            # nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)
    
    
    
class TabularEncoder_latefusion(nn.Module):
    def __init__(self, dict):
        self.input_dim = dict['input_dim']
        self.output_dim = dict['output_dim']
        
        super(TabularEncoder_latefusion, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)