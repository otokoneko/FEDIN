import torch
from torch import nn
from torch.nn import MultiheadAttention

from layers.embedding import PositionEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=8, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True, use_proj=True):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.attention = MultiheadAttention(model_dim,
                                            num_heads=num_heads,
                                            dropout=attn_dropout,
                                            batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(ffn_dim, model_dim))
        self.use_residual = use_residual
        self.dropout1 = nn.Dropout(net_dropout)
        self.dropout2 = nn.Dropout(net_dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim) if layer_norm else None
        self.layer_norm2 = nn.LayerNorm(model_dim) if layer_norm else None
        self.proj = nn.Linear(model_dim, model_dim) if use_proj else None

    def forward(self, x, attn_mask=None):
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        s = self.dropout1(attn)
        if self.use_residual:
            s += x
        if self.layer_norm1 is not None:
            s = self.layer_norm1(s)
        out = self.dropout2(self.ffn(s))
        if self.use_residual:
            out += s
        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)
        if self.proj is not None:
            out = self.proj(out)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 seq_len,
                 model_dim,
                 num_heads=8,
                 stacked_transformer_layers=1,
                 attn_dropout=0.0,
                 net_dropout=0.0,
                 use_position_emb=True,
                 layer_norm=True,
                 use_residual=True):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.transformer_blocks = nn.ModuleList(TransformerBlock(model_dim=model_dim,
                                                                 ffn_dim=model_dim,
                                                                 num_heads=num_heads,
                                                                 attn_dropout=attn_dropout,
                                                                 net_dropout=net_dropout,
                                                                 layer_norm=layer_norm,
                                                                 use_residual=use_residual)
                                                for _ in range(stacked_transformer_layers))
        self.position = PositionEmbedding(seq_len, model_dim) if use_position_emb else None

    def _get_attn_mask(self, padding_mask):
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len * self.num_heads, 1).view(-1, seq_len, seq_len)
        diag_zeros = (1 - torch.eye(seq_len, device=padding_mask.device)).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & diag_zeros
        return attn_mask

    def forward(self, x, padding_mask=None):
        # input b x len x dim
        attn_mask = None
        if padding_mask is not None:
            attn_mask = self._get_attn_mask(padding_mask)
        if self.position is not None:
            x = self.position(x)
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, attn_mask=attn_mask)
        return x
