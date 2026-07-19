import einops
from torch import nn


class Patching(nn.Module):
    def __init__(self, seq_length, patch_size):
        super(Patching, self).__init__()
        self.patch_size = patch_size

        if seq_length % patch_size != 0:
            pad_size = patch_size - seq_length % patch_size
            self.pad_layer = nn.ConstantPad1d((pad_size, 0), 0)
        else:
            self.pad_layer = nn.Identity()

    def forward(self, x):
        if x.dim() == 2:
            x = self.pad_layer(x)
            return einops.rearrange(x, "b (n p) -> (b n) p", p=self.patch_size)
        elif x.dim() == 3:
            x = self.pad_layer(x.permute(0, 2, 1))
            return einops.rearrange(x, "b d (n p) -> (b n) p d", p=self.patch_size)
        else:
            raise ValueError(f"dimension of tensor{x} should be 2 or 3")
