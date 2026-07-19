import numpy as np
import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, seq_length, model_dim):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.empty((seq_length, model_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.embedding)
        seq_len, model_dim = self.embedding.shape
        pe = torch.zeros(seq_len, model_dim)
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.embedding.data = pe

    def forward(self, x):
        length = x.shape[1]
        return x + self.embedding[-length:, :]
