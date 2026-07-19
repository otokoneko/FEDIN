import math

import einops
import torch
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, Dice
from fuxictr.pytorch.models import BaseModel
from pandas.core.common import flatten
from torch import nn

from layers.attention import TopKMultiHeadTargetAttention
from layers.complex_mlp import ComplexMLP
from layers.patching import Patching
from layers.revin import RevIN
from layers.transformer import Transformer


class FEDIN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FEDIN",
                 gpu=-1,
                 time_predictor_hidden_units=[128],
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dropout=0,
                 share_time_attention=True,
                 max_seq_length=5,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 num_heads=2,
                 router_top_k=0,
                 pred_len=3,
                 attention_alpha=1.0,
                 batch_norm=False,
                 use_mask=True,
                 target_field=[("item_id", "cate_id")],
                 sequence_field=[("click_history", "cate_history")],
                 patch_list=[5],
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_time=True,
                 use_freq=True,
                 **kwargs):
        super(FEDIN, self).__init__(feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        if not isinstance(target_field, list):
            target_field = [target_field]
        self.target_field = target_field
        if not isinstance(sequence_field, list):
            sequence_field = [sequence_field]
        self.sequence_field = sequence_field
        assert len(self.target_field) == len(self.sequence_field), \
            "len(target_field) != len(sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.use_mask = use_mask
        self.extractors = nn.ModuleList()
        patch_list = list(sorted(patch_list))
        for patch in patch_list:
            assert isinstance(patch, int) and patch > 0
        extra_dim = 0
        for target_field in self.target_field:
            input_dim = embedding_dim * len(target_field) if type(target_field) == tuple else embedding_dim
            extra_dim += input_dim
            self.extractors.append(MultiPatchExtractor(seq_length=max_seq_length,
                                                       pred_length=pred_len,
                                                       patch_list=patch_list,
                                                       num_heads=num_heads,
                                                       input_dim=input_dim,
                                                       time_predictor_hidden_units=time_predictor_hidden_units,
                                                       attention_dropout=attention_dropout,
                                                       attention_alpha=attention_alpha,
                                                       net_dropout=net_dropout,
                                                       use_time=use_time, use_freq=use_freq,
                                                       router_top_k=router_top_k,
                                                       share_time_attention=share_time_attention))
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + extra_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for idx, (target_field, sequence_field) in enumerate(zip(self.target_field,
                                                                 self.sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0]  # flatten nested list to pick the first sequence field
            mask = None
            if self.use_mask:
                mask = X[seq_field].long() != 0  # padding_idx = 0 required
            user_interest = self.extractors[idx](sequence_emb, target_emb, mask)
            for field in flatten([sequence_field, target_field]):
                feature_emb_dict.pop(field, None)
            feature_emb_dict[seq_field] = user_interest
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss += self.regularization_loss()
        return loss

    def get_embedding(self, fields, feature_emb_dict):
        if not isinstance(fields, tuple):
            fields = (fields,)
        emb_list = []
        for field in fields:
            emb = feature_emb_dict[field]
            emb_list.append(emb)

        return torch.cat(emb_list, dim=-1)


class FreqDomainTargetAttention(nn.Module):
    def __init__(self,
                 seq_length=100,
                 input_dim=64,
                 num_heads=1,
                 dropout_rate=0.,
                 attention_alpha=1.,
                 use_scale=True):
        super(FreqDomainTargetAttention, self).__init__()
        attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.attention_alpha = attention_alpha

    def forward(self, x, target):
        query, key, value = target, x, x

        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        scores = torch.matmul(query, key.transpose(-1, -2))
        if self.scale:
            scores = scores / self.scale
        scores = abs(torch.fft.rfft(scores, dim=-1))

        attention = scores.softmax(dim=-1) * self.attention_alpha
        if self.dropout is not None:
            attention = self.dropout(attention)

        fft_s = (value.size(-2), value.size(-1))
        value = torch.fft.rfft2(value)
        value = value * (1 + attention)
        value = torch.fft.irfft2(value, s=fft_s)
        value = einops.rearrange(value, "b h d l -> b l (h d)")
        attention = attention.repeat(1, 1, self.head_dim, 1)
        attention = einops.rearrange(attention, "b h d l -> b (h d) l")
        return value, attention


class FreqDomainExtractor(nn.Module):
    def __init__(self, seq_length, pred_length, input_dim, num_heads, net_dropout, attention_dropout, attention_alpha):
        super(FreqDomainExtractor, self).__init__()
        self.pred_len = pred_length
        self.out_len = seq_length + pred_length
        self.pad = nn.ConstantPad1d((0, pred_length), 0)
        freq_num = (self.out_len // 2) + 1
        self.predict = ComplexMLP(freq_num,
                                  hidden_units=[128, 128],
                                  output_dim=freq_num,
                                  dropout_rates=net_dropout)
        self.attention = FreqDomainTargetAttention(seq_length=seq_length,
                                                   input_dim=input_dim,
                                                   num_heads=num_heads,
                                                   attention_alpha=attention_alpha,
                                                   dropout_rate=attention_dropout)

    def forward(self, x, target, mask):
        x, attention = self.attention(x, target)
        x = x.permute(0, 2, 1)
        x = self.pad(x)

        b, c, l = x.shape
        x_fft = torch.fft.rfft2(x)
        x_fft = self.predict(x_fft)
        out = torch.fft.irfft2(x_fft, s=(c, l))
        out = out[..., -self.pred_len:]
        return out.permute(0, 2, 1), attention

class TimeFreqScaler(nn.Module):
    def __init__(self, seq_length, net_dropout):
        super(TimeFreqScaler, self).__init__()
        freq_num = (seq_length // 2) + 1
        self.mlp = MLP_Block(freq_num,
                             hidden_units=[64, 64],
                             output_dim=2,
                             output_activation='softmax',
                             dropout_rates=net_dropout,
                             use_bias=False)

    def forward(self, attention, time_emb, freq_emb):
        weight = self.mlp(attention)
        weight = weight.unsqueeze(1)
        out = torch.stack([time_emb, freq_emb], dim=-1) * weight
        out = out.sum(dim=-1)
        return out


class TimeDomainExtractor(nn.Module):
    def __init__(self, seq_length, pred_length, patch_size, predictor_hidden_units, input_dim,
                 net_dropout, attention_dropout, num_heads, attention):
        super(TimeDomainExtractor, self).__init__()
        self.patch_size = patch_size
        self.patch_num = int(math.ceil(seq_length / patch_size))
        self.patching = Patching(seq_length, patch_size)
        if attention is not None:
            self.attention = attention
        else:
            self.attention = Transformer(patch_size,
                                         model_dim=input_dim,
                                         attn_dropout=attention_dropout,
                                         net_dropout=attention_dropout,
                                         num_heads=num_heads)
        self.mlp = MLP_Block(input_dim=self.patch_num,
                             hidden_units=predictor_hidden_units,
                             output_dim=pred_length,
                             dropout_rates=net_dropout)

    def forward(self, x, target, mask):
        batch, seq_len, hidden = x.shape
        score = (x * target.unsqueeze(1)).sum(dim=-1, keepdim=True) / (hidden ** 0.5)
        x = score.softmax(dim=1) * x
        x = self.patching(x).contiguous()
        mask_pad = None
        if mask is not None:
            mask_pad = self.patching(mask)
        out = self.attention(x, ~mask_pad)
        out = out.sum(dim=1)
        out = out.reshape(batch, -1, hidden)
        out = self.mlp(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class SinglePatchExtractor(nn.Module):
    def __init__(self, seq_length, pred_length, patch_size, num_heads, input_dim,
                 time_predictor_hidden_units, attention_dropout, attention_alpha,
                 net_dropout, use_time, use_freq, time_attention=None):
        super(SinglePatchExtractor, self).__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.revin = RevIN(input_dim)
        self.freq_extractor = FreqDomainExtractor(seq_length=seq_length,
                                                  pred_length=pred_length,
                                                  input_dim=input_dim,
                                                  num_heads=num_heads,
                                                  net_dropout=net_dropout,
                                                  attention_dropout=attention_dropout,
                                                  attention_alpha=attention_alpha)
        self.time_extractor = TimeDomainExtractor(seq_length=seq_length,
                                                  pred_length=pred_length,
                                                  patch_size=patch_size,
                                                  predictor_hidden_units=time_predictor_hidden_units,
                                                  net_dropout=net_dropout,
                                                  attention_dropout=attention_dropout,
                                                  num_heads=num_heads,
                                                  input_dim=input_dim,
                                                  attention=time_attention)
        self.use_time = use_time
        self.use_freq = use_freq
        self.scaler = TimeFreqScaler(seq_length, net_dropout)

    def forward(self, x, target, mask):
        x = self.revin(torch.cat([x, target.unsqueeze(1)], dim=1), 'norm')
        target, x = x[:, -1, :], x[:, :-1, :]
        if self.use_time:
            time_emb = self.time_extractor(x, target, mask)
        if self.use_freq:
            freq_emb, attention = self.freq_extractor(x, target, mask)
        if self.use_freq and self.use_time:
            out = self.scaler(attention, time_emb, freq_emb)
        elif self.use_time:
            out = time_emb
        elif self.use_freq:
            out = freq_emb
        out = self.revin(out, 'denorm')
        return out


class MultiPatchExtractor(nn.Module):
    def __init__(self, seq_length, pred_length, patch_list, num_heads, input_dim, share_time_attention,
                 time_predictor_hidden_units, attention_dropout, attention_alpha, net_dropout, router_top_k,
                 use_time, use_freq):
        super(MultiPatchExtractor, self).__init__()
        self.num_heads = num_heads
        self.patch_list = patch_list
        self.model_dim = input_dim
        time_attention = Transformer(max(patch_list),
                                     model_dim=input_dim,
                                     attn_dropout=attention_dropout,
                                     net_dropout=attention_dropout,
                                     num_heads=num_heads) if share_time_attention else None
        self.extractors = nn.ModuleList([
            SinglePatchExtractor(seq_length=seq_length,
                                 pred_length=pred_length,
                                 patch_size=p,
                                 num_heads=num_heads,
                                 input_dim=input_dim,
                                 time_predictor_hidden_units=time_predictor_hidden_units,
                                 attention_dropout=attention_dropout,
                                 attention_alpha=attention_alpha,
                                 net_dropout=net_dropout,
                                 time_attention=time_attention,
                                 use_time=use_time,
                                 use_freq=use_freq)
            for p in patch_list
        ])
        self.router = TopKMultiHeadTargetAttention(input_dim,
                                                   attention_dim=input_dim,
                                                   dropout_rate=attention_dropout,
                                                   num_heads=1,
                                                   topk=router_top_k,
                                                   use_qkvo=False)

    def forward(self, x, target, mask):
        result = []
        for i, patch_size in enumerate(self.patch_list):
            emb = self.extractors[i](x, target, mask)
            result.append(emb)
        result = torch.cat(result, dim=1)
        result = self.router(target, result)
        return torch.cat([result, result * target, target], dim=-1)
