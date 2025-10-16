import torch
import torch.nn as nn

class MLPAdapter(nn.Module):
    def __init__(self, in_dim=512, out_dim=2818, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)