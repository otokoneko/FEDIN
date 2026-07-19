from complexNN.nn import cLinear, cRelu, cDropout
from torch import nn


class ComplexMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 output_dim=None,
                 dropout_rates=0.0,
                 use_bias=True):
        super(ComplexMLP, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(cLinear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            dense_layers.append(cRelu())
            if dropout_rates[idx] > 0:
                dense_layers.append(cDropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(cLinear(hidden_units[-1], output_dim, bias=use_bias))
        self.mlp = nn.Sequential(*dense_layers)

    def forward(self, x):
        return self.mlp(x)
