import torch

from torch import nn


class LinearGenerator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(noise_dim, output_dim))
        self.b = nn.Parameter(2 * torch.randn(output_dim))

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        return torch.add(torch.matmul(self.W, z.unsqueeze(2)).squeeze(), self.b)


class LinearDualVariable(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.v = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        return torch.matmul(self.v.T, x.unsqueeze(2))

    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function"""
        with torch.no_grad():
            # Normalize vector
            self.v = nn.Parameter(self.v / torch.norm(self.v, p=2))

