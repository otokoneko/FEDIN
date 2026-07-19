# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import math

import numpy as np
import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from torch.nn import MultiheadAttention


class SASRec(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="SASRec",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=32,
                 max_sequence_len=430,
                 num_heads=1,
                 batch_norm=True,
                 target_field=[("item_id", "cate_id")],
                 sequence_field=[("click_history", "cate_history")],
                 seq_pooling_type="mean",  # ["mean", "sum", "target", "concat"]
                 use_causal_mask=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0.0,
                 attention_dropout=0.0,
                 dnn_hidden_units=[64, 32],
                 dnn_hidden_activations='sigmoid',
                 **kwargs):
        super(SASRec, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.max_sequence_len = max_sequence_len
        self.seq_pooling_type = seq_pooling_type
        self.use_causal_mask = use_causal_mask
        self.num_heads = num_heads
        if not isinstance(target_field, list):
            target_field = [target_field]
        self.target_field = target_field
        if not isinstance(sequence_field, list):
            sequence_field = [sequence_field]
        self.sequence_field = sequence_field
        assert len(self.target_field) == len(self.sequence_field), \
            "len(target_field) != len(sequence_field)"

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.position_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        extra_dim = 0
        for target_field in self.target_field:
            input_dim = embedding_dim * len(target_field) if isinstance(target_field, tuple) else embedding_dim
            extra_dim += input_dim
            self.position_layers.append(nn.Embedding(self.max_sequence_len + 1, input_dim))
            self.attention_layers.append(
                SASRecAttention(model_dim=input_dim,
                                ffn_dim=input_dim,
                                num_heads=num_heads,
                                attn_dropout=attention_dropout,
                                net_dropout=net_dropout,
                                layer_norm=True,
                                use_residual=True))
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + extra_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for target_field, sequence_field, attention_layer, position_layer in zip(self.target_field,
                                                                                 self.sequence_field,
                                                                                 self.attention_layers,
                                                                                 self.position_layers):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            target_pos = torch.zeros([target_emb.size(0)], dtype=torch.int, device=self.device)
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            sequence_pos = (torch.arange(1, sequence_emb.size(1) + 1, 1, dtype=torch.int, device=self.device)
                            .unsqueeze(0).expand(sequence_emb.size(0), -1))

            target_emb = target_emb + position_layer(target_pos)
            sequence_emb = sequence_emb + position_layer(sequence_pos)

            seq_field = list(flatten([sequence_field]))[0]  # flatten nested list to pick the first field
            padding_mask, attn_mask = self.get_mask(X[seq_field])

            transformer_out = attention_layer(sequence_emb, attn_mask)
            pooling_emb = self.sequence_pooling(transformer_out, padding_mask)
            for sequence, target, field_emb, tar_emb in zip(list(flatten([sequence_field])),
                                                            list(flatten([target_field])),
                                                            pooling_emb.split(self.embedding_dim, dim=-1),
                                                            target_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[sequence] = torch.cat([field_emb, field_emb * tar_emb], dim=-1)
                feature_emb_dict[target] = tar_emb

        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_mask(self, x):
        """ padding_mask: 1 for masked positions
            attn_mask: 1 for masked positions in nn.MultiheadAttention
        """
        padding_mask = (x == 0)
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len * self.num_heads, 1).view(-1, seq_len, seq_len)
        diag_zeros = (1 - torch.eye(seq_len, device=x.device)).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & diag_zeros
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool() \
                .unsqueeze(0).expand_as(attn_mask)
            attn_mask = attn_mask | causal_mask
        return padding_mask, attn_mask

    def sequence_pooling(self, transformer_out, mask):
        mask = (1 - mask.float()).unsqueeze(-1)  # 0 for masked positions
        if self.seq_pooling_type == "mean":
            return (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1.e-12)
        elif self.seq_pooling_type == "sum":
            return (transformer_out * mask).sum(dim=1)
        elif self.seq_pooling_type == "target":
            return transformer_out[:, -1, :]
        elif self.seq_pooling_type == "concat":
            return transformer_out.flatten(start_dim=1)
        else:
            raise ValueError("seq_pooling_type={} not supported.".format(self.seq_pooling_type))


class SASRecAttention(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=8, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super(SASRecAttention, self).__init__()
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
        return out

